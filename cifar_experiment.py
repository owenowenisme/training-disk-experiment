import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader, Dataset
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import torch.nn as nn
import torch.optim as optim
import time
import os
import numpy as np

class CIFARDataset(Dataset):
    """Custom dataset wrapper for CIFAR data"""
    def __init__(self, data_tensors, labels):
        self.data = data_tensors
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'image': self.data[idx],
            'label': self.labels[idx]
        }

def create_cifar_imbalanced_datasets():
    """Create CIFAR-10 with imbalanced distribution across nodes"""
    
    print("Downloading and preparing CIFAR-10 dataset...")
    
    # Create data directory
    os.makedirs('cifar_data', exist_ok=True)
    
    # Download CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_dataset = torchvision.datasets.CIFAR10(
        root='./cifar_data', train=True, download=True, transform=transform
    )
    
    print(f"Full CIFAR-10 dataset size: {len(full_dataset)}")
    
    # Create imbalanced splits
    distributions = {
        "equal": [16000, 16000, 18000],        # Equal distribution
        "imbalanced": [30000, 15000, 5000],    # 6:3:1 ratio
        "extreme": [40000, 8000, 2000]         # 20:4:1 ratio
    }
    
    created_datasets = {}
    
    for exp_name, sizes in distributions.items():
        print(f"\nCreating {exp_name} distribution: {sizes}")
        
        # Ensure sizes don't exceed dataset size
        total_requested = sum(sizes)
        if total_requested > len(full_dataset):
            # Scale down proportionally
            scale_factor = len(full_dataset) / total_requested
            sizes = [int(s * scale_factor) for s in sizes]
            # Adjust last element to match exact total
            sizes[-1] = len(full_dataset) - sum(sizes[:-1])
            print(f"Adjusted sizes: {sizes}")
        
        # Split dataset according to sizes
        splits = random_split(full_dataset, sizes)
        
        # Convert to Ray datasets and save
        exp_dir = f'cifar_data/{exp_name}'
        os.makedirs(exp_dir, exist_ok=True)
        
        node_datasets = []
        for i, split in enumerate(splits):
            # Extract data and labels from the split
            data_list = []
            labels_list = []
            
            for idx in range(len(split)):
                sample = split[idx]
                data_list.append(sample[0])  # Image tensor
                labels_list.append(sample[1])  # Label
            
            # Convert to Ray dataset format
            ray_data = []
            for j in range(len(data_list)):
                ray_data.append({
                    'image': data_list[j].numpy(),
                    'label': labels_list[j]
                })
            
            # Create Ray dataset
            node_dataset = ray.data.from_items(ray_data)
            node_datasets.append(node_dataset)
            
            print(f"  Node {i}: {len(ray_data)} samples")
        
        created_datasets[exp_name] = node_datasets
    
    return created_datasets, distributions

def load_cifar_experiment_data(experiment_type, datasets_dict):
    """Load the specific experiment data"""
    if experiment_type not in datasets_dict:
        raise ValueError(f"Experiment type {experiment_type} not found")
    
    node_datasets = datasets_dict[experiment_type]
    
    # Combine all node datasets into one for Ray to distribute
    combined_dataset = node_datasets[0]
    for ds in node_datasets[1:]:
        combined_dataset = combined_dataset.union(ds)
    
    return combined_dataset

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

def preprocess_cifar_batch(batch):
    """Preprocess CIFAR batch for training"""
    # Convert numpy arrays back to tensors
    images = torch.stack([torch.from_numpy(img) for img in batch['image']])
    labels = torch.tensor(batch['label'])
    
    return {
        'image': images,
        'label': labels
    }

def train_func_cifar():
    """Training function for CIFAR-10 experiment"""
    worker_rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    
    print(f"Worker {worker_rank}: Starting CIFAR training")
    
    batch_size = 64
    
    # Get the dataset shard for this worker (returns DataIterator)
    train_data_shard = train.get_dataset_shard("train")
    print(f"Worker {worker_rank}: Got data shard (samples will be counted during iteration)")
    
    # Create data loader directly from DataIterator - NO map_batches!
    train_dataloader = train_data_shard.iter_torch_batches(
        batch_size=batch_size,
        dtypes={'image': torch.float32, 'label': torch.long}
    )
    
    # Model setup
    model = ResNet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    start_time = time.time()
    total_batches = 0
    total_samples = 0
    
    # Training loop
    for epoch in range(5):
        epoch_start = time.time()
        epoch_batches = 0
        epoch_samples = 0
        running_loss = 0.0
        
        print(f"Worker {worker_rank}: Starting epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Handle preprocessing in the loop
            inputs = batch['image']
            labels = batch['label']
            
            # Ensure correct tensor types
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            epoch_batches += 1
            total_batches += 1
            batch_size_actual = inputs.shape[0]
            epoch_samples += batch_size_actual
            total_samples += batch_size_actual
            
            if batch_idx % 50 == 0:
                print(f"Worker {worker_rank}: Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / epoch_batches if epoch_batches > 0 else 0
        
        print(f"Worker {worker_rank}: Epoch {epoch+1} completed - {epoch_samples} samples, {epoch_time:.2f}s, avg_loss: {avg_loss:.4f}")
        
        train.report({
            f"worker_{worker_rank}_epoch_{epoch}_time": epoch_time,
            f"worker_{worker_rank}_epoch_{epoch}_samples": epoch_samples,
            f"worker_{worker_rank}_epoch_{epoch}_batches": epoch_batches,
            f"worker_{worker_rank}_epoch_{epoch}_loss": avg_loss
        })
    
    total_time = time.time() - start_time
    print(f"Worker {worker_rank}: Training completed - {total_samples} samples, {total_time:.2f}s")
    
    train.report({
        f"worker_{worker_rank}_total_time": total_time,
        f"worker_{worker_rank}_total_samples": total_samples,
        f"worker_{worker_rank}_total_batches": total_batches,
        f"worker_{worker_rank}_samples_per_second": total_samples / total_time if total_time > 0 else 0
    })

def run_cifar_experiment(experiment_type, datasets_dict, distributions):
    """Run CIFAR experiment with specified data distribution"""
    print(f"\n{'='*60}")
    print(f"Running CIFAR-10 experiment: {experiment_type}")
    print(f"Data distribution: {distributions[experiment_type]}")
    print(f"{'='*60}")
    
    # Load experiment data
    dataset = load_cifar_experiment_data(experiment_type, datasets_dict)
    
    # Create trainer
    trainer = TorchTrainer(
        train_func_cifar,
        datasets={"train": dataset},
        scaling_config=ScalingConfig(
            num_workers=15,  # 3 workers to match the 3 data splits
            use_gpu=False   # Set to True if you have GPUs
        )
    )
    
    start_time = time.time()
    result = trainer.fit()
    total_time = time.time() - start_time
    
    print(f"Total experiment time: {total_time:.2f}s")
    
    # Analyze results
    analyze_cifar_results(result, experiment_type, total_time, distributions[experiment_type])
    
    return result

def analyze_cifar_results(result, experiment_type, total_time, distribution):
    """Analyze results from CIFAR experiment"""
    metrics = result.metrics
    
    print(f"\n--- Results for {experiment_type} ---")
    print(f"Data distribution: {distribution}")
    print(f"Total experiment time: {total_time:.2f}s")
    
    worker_data = []
    
    # Extract worker metrics
    for i in range(3):
        worker_info = {
            'worker': i,
            'total_time': metrics.get(f"worker_{i}_total_time", 0),
            'total_samples': metrics.get(f"worker_{i}_total_samples", 0),
            'samples_per_second': metrics.get(f"worker_{i}_samples_per_second", 0)
        }
        worker_data.append(worker_info)
    
    # Print worker statistics
    print("\nWorker Performance:")
    for worker in worker_data:
        if worker['total_time'] > 0:
            print(f"  Worker {worker['worker']}: {worker['total_samples']:,} samples, "
                  f"{worker['total_time']:.2f}s, "
                  f"{worker['samples_per_second']:.1f} samples/sec")
    
    # Calculate imbalance metrics
    times = [w['total_time'] for w in worker_data if w['total_time'] > 0]
    samples = [w['total_samples'] for w in worker_data if w['total_samples'] > 0]
    
    if times and len(times) > 1:
        print(f"\nImbalance Analysis:")
        print(f"  Time difference: {max(times) - min(times):.2f}s")
        print(f"  Sample distribution: {samples}")
        print(f"  Training efficiency: {min(times)/max(times)*100:.1f}%")
        print(f"  Stragglers overhead: {(max(times) - min(times))/max(times)*100:.1f}%")

def main():
    """Main function to run CIFAR experiments"""
    ray.init(ignore_reinit_error=True)
    
    try:
        # Step 1: Create imbalanced datasets
        print("Step 1: Creating imbalanced CIFAR-10 datasets...")
        datasets_dict, distributions = create_cifar_imbalanced_datasets()
        print(datasets_dict.keys())
        print(distributions.keys())
        
        # Step 2: Run experiments
        print("\nStep 2: Running experiments...")
        results = {}
        
        for exp_type in ["equal", "imbalanced", "extreme"]:
            try:
                result = run_cifar_experiment(exp_type, datasets_dict, distributions)
                results[exp_type] = result
                print(f"✓ {exp_type} experiment completed successfully")
            except Exception as e:
                print(f"✗ Error in {exp_type} experiment: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 3: Comparative analysis
        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS")
        print("="*60)
        
        for exp_type, result in results.items():
            distribution = distributions[exp_type]
            print(f"\n{exp_type.upper()}: {distribution}")
            
            # Extract key metrics for comparison
            metrics = result.metrics
            worker_times = []
            for i in range(3):
                time_key = f"worker_{i}_total_time"
                if time_key in metrics:
                    worker_times.append(metrics[time_key])
            
            if worker_times:
                efficiency = min(worker_times)/max(worker_times)*100
                print(f"  Efficiency: {efficiency:.1f}%")
                print(f"  Time spread: {max(worker_times) - min(worker_times):.2f}s")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main() 