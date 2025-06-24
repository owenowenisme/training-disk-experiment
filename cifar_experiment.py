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

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

def train_func_worker_specific(config):
    """Training function where each worker creates only its portion of data locally"""
    
    worker_rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    
    print(f"Worker {worker_rank}: Starting worker-specific CIFAR training")
    
    # Get experiment type from config instead of environment
    if 'experiment_type' not in config:
        raise ValueError("Experiment type not specified in config")
    experiment_type = config['experiment_type']
    
    # Define distributions for TRUE imbalance
    distributions = {
        "equal": [16000, 16000, 18000],        # Equal distribution
        "imbalanced": [30000, 15000, 5000],    # 6:3:1 ratio
        "extreme": [40000, 8000, 2000]         # 20:4:1 ratio
    }
    
    worker_sizes = distributions[experiment_type]
    my_data_size = worker_sizes[worker_rank] if worker_rank < len(worker_sizes) else 1000
    
    print(f"Worker {worker_rank}: Will create {my_data_size} samples for {experiment_type}")
    
    # Each worker downloads CIFAR-10 to its own temp directory
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        full_dataset = torchvision.datasets.CIFAR10(
            root=f'/tmp/cifar_data_worker_{worker_rank}', 
            train=True, 
            download=True, 
            transform=transform
        )
        print(f"Worker {worker_rank}: Downloaded CIFAR-10 with {len(full_dataset)} samples")
    except Exception as e:
        print(f"Worker {worker_rank}: Error downloading CIFAR-10: {e}")
        # Fallback to synthetic data
        print(f"Worker {worker_rank}: Using synthetic data instead")
        worker_data = []
        for i in range(my_data_size):
            worker_data.append({
                'image': np.random.randn(3, 32, 32).astype(np.float32),
                'label': i % 10
            })
        
        worker_dataset = ray.data.from_items(worker_data)
        train_dataloader = worker_dataset.iter_torch_batches(
            batch_size=64,
            dtypes={'image': torch.float32, 'label': torch.long}
        )
        
        # Continue with synthetic data...
        full_dataset = None
    
    if full_dataset is not None:
        # Create worker-specific data by deterministic sampling
        # Use worker_rank as seed to ensure different but reproducible data per worker
        import random
        random.seed(worker_rank * 1000 + hash(experiment_type))  # Different seed per worker per experiment
        
        # Sample exactly the amount this worker should process
        available_samples = len(full_dataset)
        actual_size = min(my_data_size, available_samples)
        
        if actual_size < available_samples:
            indices = random.sample(range(available_samples), actual_size)
        else:
            indices = list(range(available_samples))
        
        print(f"Worker {worker_rank}: Sampling {len(indices)} indices from {available_samples} available")
        
        worker_data = []
        for idx in indices:
            image, label = full_dataset[idx]
            worker_data.append({
                'image': image.numpy(),
                'label': int(label)
            })
        
        print(f"Worker {worker_rank}: Created {len(worker_data)} samples")
        
        # Convert to Ray dataset for this worker only
        worker_dataset = ray.data.from_items(worker_data)
        
        # Create dataloader
        train_dataloader = worker_dataset.iter_torch_batches(
            batch_size=64,
            dtypes={'image': torch.float32, 'label': torch.long}
        )
    
    # Model setup
    model = ResNet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    start_time = time.time()
    total_batches = 0
    total_samples = 0
    
    # Training loop - fewer epochs for experiment
    for epoch in range(3):
        epoch_start = time.time()
        epoch_batches = 0
        epoch_samples = 0
        running_loss = 0.0
        
        print(f"Worker {worker_rank}: Starting epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(train_dataloader):
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
            epoch_samples += inputs.shape[0]
            total_samples += inputs.shape[0]
            
            if batch_idx % 100 == 0:
                print(f"Worker {worker_rank}: Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / epoch_batches if epoch_batches > 0 else 0
        
        print(f"Worker {worker_rank}: Epoch {epoch+1} completed - {epoch_samples} samples, {epoch_time:.2f}s")
        
        # Report metrics
        train.report({
            f"worker_{worker_rank}_epoch_{epoch}_time": epoch_time,
            f"worker_{worker_rank}_epoch_{epoch}_samples": epoch_samples,
            f"worker_{worker_rank}_epoch_{epoch}_loss": avg_loss
        })
    
    total_time = time.time() - start_time
    
    print(f"Worker {worker_rank}: Training completed - {total_samples} samples in {total_time:.2f}s")
    
    # Final report
    train.report({
        f"worker_{worker_rank}_total_time": total_time,
        f"worker_{worker_rank}_total_samples": total_samples,
        f"worker_{worker_rank}_total_batches": total_batches,
        f"worker_{worker_rank}_samples_per_second": total_samples / total_time if total_time > 0 else 0
    })

def run_worker_specific_experiment(experiment_type, distributions):
    """Run experiment with TRUE worker-specific data distribution"""
    
    print(f"\n{'='*60}")
    print(f"Running Worker-Local CIFAR-10 experiment: {experiment_type}")
    print(f"TRUE data distribution: {distributions[experiment_type]}")
    print(f"{'='*60}")
    
    # Remove environment variable - pass through config instead
    # os.environ['EXPERIMENT_TYPE'] = experiment_type
    
    # Create trainer with train_loop_config
    trainer = TorchTrainer(
        train_func_worker_specific,
        train_loop_config={'experiment_type': experiment_type},  # Pass through config
        scaling_config=ScalingConfig(
            num_workers=3,
            use_gpu=False,
            resources_per_worker={"CPU": 4},
        )
    )
    
    start_time = time.time()
    result = trainer.fit()
    total_time = time.time() - start_time
    
    print(f"Total experiment time: {total_time:.2f}s")
    
    # Analyze true imbalance results
    analyze_worker_specific_results(result, experiment_type, total_time, distributions[experiment_type])
    return result

def analyze_worker_specific_results(result, experiment_type, total_time, true_distribution):
    """Analyze results showing TRUE data imbalance effects"""
    
    metrics = result.metrics
    
    print(f"\n--- TRUE IMBALANCE Results for {experiment_type} ---")
    print(f"Intended distribution: {true_distribution}")
    print(f"Total experiment time: {total_time:.2f}s")
    
    worker_data = []
    actual_samples = []
    
    for i in range(3):
        worker_info = {
            'worker': i,
            'total_time': metrics.get(f"worker_{i}_total_time", 0),
            'total_samples': metrics.get(f"worker_{i}_total_samples", 0),
            'samples_per_second': metrics.get(f"worker_{i}_samples_per_second", 0),
            'intended_samples': true_distribution[i] if i < len(true_distribution) else 0
        }
        worker_data.append(worker_info)
        actual_samples.append(worker_info['total_samples'])
    
    print(f"Actual sample distribution: {actual_samples}")
    
    # Verify true imbalance was achieved
    print("\nWorker Performance (TRUE IMBALANCE):")
    for worker in worker_data:
        if worker['total_time'] > 0:
            print(f"  Worker {worker['worker']}: {worker['total_samples']:,} samples "
                  f"(intended: {worker['intended_samples']:,}), "
                  f"{worker['total_time']:.2f}s, "
                  f"{worker['samples_per_second']:.1f} samples/sec")
    
    # Calculate TRUE imbalance impact
    times = [w['total_time'] for w in worker_data if w['total_time'] > 0]
    samples = [w['total_samples'] for w in worker_data if w['total_samples'] > 0]
    
    if times and len(times) > 1:
        print(f"\nTRUE Imbalance Analysis:")
        print(f"  Time difference: {max(times) - min(times):.2f}s")
        print(f"  Slowest worker: {max(times):.2f}s")
        print(f"  Fastest worker: {min(times):.2f}s")
        print(f"  Training efficiency: {min(times)/max(times)*100:.1f}%")
        print(f"  Stragglers overhead: {(max(times) - min(times))/max(times)*100:.1f}%")
        
        # Calculate theoretical vs actual speedup loss
        if all(times) and all(samples):
            total_samples = sum(samples)
            total_throughput = sum([s/t for s, t in zip(samples, times)])
            ideal_time = total_samples / total_throughput if total_throughput > 0 else 0
            actual_time = max(times)
            speedup_loss = (actual_time - ideal_time) / ideal_time * 100 if ideal_time > 0 else 0
            print(f"  Speedup loss due to imbalance: {speedup_loss:.1f}%")

def main():
    """Main function for worker-local TRUE imbalance experiment"""
    
    ray.init(ignore_reinit_error=True)
    
    try:
        # Define distributions (no file creation needed - workers create data locally)
        distributions = {
            "equal": [16000, 16000, 18000],        # Equal distribution
            "imbalanced": [30000, 15000, 5000],    # 6:3:1 ratio
            "extreme": [40000, 8000, 2000]         # 20:4:1 ratio
        }
        
        print(f"Defined distributions: {list(distributions.keys())}")
        print("Workers will create data locally - no file distribution needed!")
        
        # Run experiments with TRUE imbalance
        print("\nStep 1: Running TRUE imbalance experiments...")
        results = {}
        
        for exp_type in ["equal", "imbalanced", "extreme"]:
            try:
                result = run_worker_specific_experiment(exp_type, distributions)
                results[exp_type] = result
                print(f"✓ {exp_type} experiment completed successfully")
            except Exception as e:
                print(f"✗ Error in {exp_type} experiment: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 2: Comparative analysis showing TRUE imbalance effects
        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS - TRUE DATA IMBALANCE")
        print("="*60)
        
        for exp_type, result in results.items():
            distribution = distributions[exp_type]
            print(f"\n{exp_type.upper()}: {distribution}")
            
            metrics = result.metrics
            worker_times = []
            worker_samples = []
            
            for i in range(3):
                time_key = f"worker_{i}_total_time"
                samples_key = f"worker_{i}_total_samples"
                if time_key in metrics:
                    worker_times.append(metrics[time_key])
                    worker_samples.append(metrics[samples_key])
            
            if worker_times:
                efficiency = min(worker_times)/max(worker_times)*100
                print(f"  Efficiency: {efficiency:.1f}%")
                print(f"  Time spread: {max(worker_times) - min(worker_times):.2f}s")
                print(f"  Sample spread: {worker_samples}")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()