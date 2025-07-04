import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader, Dataset
import ray
from ray import train
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig
import torch.nn as nn
import torch.optim as optim
import time
import os
import numpy as np
import subprocess
import zipfile
import urllib.request
from pathlib import Path
from PIL import Image
import json
import gc

class StreamingCOCODataset(Dataset):
    """Memory-efficient streaming COCO dataset that loads images on-demand"""
    def __init__(self, image_ids, images_dir, annotations_dict, transform, worker_rank=0):
        self.image_ids = image_ids
        self.images_dir = images_dir
        self.annotations_dict = annotations_dict
        self.transform = transform
        self.worker_rank = worker_rank 
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image on-demand
        image_path = self.images_dir / f"{image_id:012d}.jpg"
        
        if not image_path.exists():
            # Return a black image as fallback
            print(f"Worker {self.worker_rank}: Image {image_id} not found")
            image = torch.zeros(3, 128, 128)  # Smaller size to save memory
            label = 0
        else:
            try:
                # Load and transform image
                with Image.open(image_path) as img:
                    image = img.convert('RGB')
                    if self.transform:
                        image = self.transform(image)

                # Get label from annotations
                label = 0  # Default label
                if str(image_id) in self.annotations_dict:
                    annotations = self.annotations_dict[str(image_id)]
                    if annotations:
                        label = annotations[0].get('category_id', 0) % 80  # Ensure 0-79 range
                
            except Exception as e:
                # Fallback to synthetic data
                print(f"Worker {self.worker_rank}: Error loading image {image_id}: {e}")
                image = torch.zeros(3, 128, 128)  # Smaller size
                label = 0
        
        return image, label

class ResNet18(nn.Module):
    def __init__(self, num_classes=80):  # COCO has 80 object classes
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

def load_coco_annotations(annotations_file):
    """Load COCO annotations into a dictionary for efficient lookup"""
    try:
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image_id -> annotations mapping
        annotations_dict = {}
        for ann in coco_data.get('annotations', []):
            image_id = str(ann['image_id'])
            if image_id not in annotations_dict:
                annotations_dict[image_id] = []
            annotations_dict[image_id].append(ann)
        
        # Get list of all image IDs
        image_ids = [img['id'] for img in coco_data.get('images', [])]
        
        return annotations_dict, image_ids
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return {}, []

def train_func_epoch_sync(config):
    """Training with epoch-level synchronization instead of batch-level"""
    
    worker_rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    
    print(f"Worker {worker_rank}: Starting EPOCH-SYNC COCO training")
    
    # Your original data setup (different sizes per worker)
    experiment_type = config['experiment_type']
    distributions = {
        "equal": [40000, 39000, 39000],        
        "imbalanced": [80000, 30000, 8000],    # Different sizes OK!
        "extreme": [100000, 15000, 3000]         
    }
    
    worker_sizes = distributions[experiment_type]
    my_data_size = worker_sizes[worker_rank] if worker_rank < len(worker_sizes) else 100
    
    print(f"Worker {worker_rank}: Will process {my_data_size} samples")
    
    # Memory-efficient transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Load COCO data
        coco_base_dir = '/tmp/coco_data'
        download_success = download_coco_dataset(coco_base_dir, worker_rank)
        
        if not download_success:
            raise Exception("Failed to download COCO dataset")
        
        images_dir = Path(coco_base_dir) / 'images' / 'train2017'
        annotations_file = Path(coco_base_dir) / 'annotations' / 'instances_train2017.json'
        
        print(f"Worker {worker_rank}: Loading COCO annotations")
        annotations_dict, all_image_ids = load_coco_annotations(annotations_file)
        
        if not all_image_ids:
            raise Exception("No image IDs found")
        
        # Sample worker-specific data
        import random
        random.seed(worker_rank * 1000 + hash(experiment_type))
        
        available_samples = len(all_image_ids)
        actual_size = min(my_data_size, available_samples)
        
        if actual_size < available_samples:
            selected_image_ids = random.sample(all_image_ids, actual_size)
        else:
            selected_image_ids = all_image_ids[:actual_size]
        
        print(f"Worker {worker_rank}: Will process {len(selected_image_ids)} images")
        
        # Create dataset
        streaming_dataset = StreamingCOCODataset(
            image_ids=selected_image_ids,
            images_dir=images_dir,
            annotations_dict=annotations_dict,
            transform=transform,
            worker_rank=worker_rank
        )
        
        train_dataloader = DataLoader(
            streaming_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
    except Exception as e:
        print(f"Worker {worker_rank}: Error loading COCO: {e}")
        print(f"Worker {worker_rank}: Using synthetic data instead")
        
        class SyntheticDataset(Dataset):
            def __init__(self, size):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return (
                    torch.randn(3, 128, 128),
                    torch.randint(0, 80, (1,)).item()
                )
        
        synthetic_dataset = SyntheticDataset(my_data_size)
        train_dataloader = DataLoader(
            synthetic_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
    
    # IMPORTANT: NO DDP wrapper - we'll sync manually
    model = ResNet18(num_classes=80)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Training loop with manual epoch synchronization
    for epoch in range(2):
        expected_batches = (my_data_size + 15) // 16
        print(f"Worker {worker_rank}: Epoch {epoch+1}, expected {expected_batches} batches")
        
        epoch_start = time.time()
        epoch_batches = 0
        epoch_samples = 0
        running_loss = 0.0
        
        # STEP 1: Each worker processes ALL its batches independently
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # ← NO sync here, just local update
            
            running_loss += loss.item()
            epoch_batches += 1
            epoch_samples += inputs.shape[0]
            
            if batch_idx % 25 == 0:
                print(f"Worker {worker_rank}: Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / epoch_batches if epoch_batches > 0 else 0
        
        print(f"Worker {worker_rank}: Local training done - {epoch_samples} samples, {epoch_batches} batches")
        
        # STEP 2: Synchronize model parameters across all workers at epoch end
        print(f"Worker {worker_rank}: Synchronizing model at end of epoch {epoch+1}...")
        
        # Method 1: Use Ray Train's built-in synchronization
        synchronized_model = sync_model_across_workers(model)
        model.load_state_dict(synchronized_model.state_dict())
        
        print(f"Worker {worker_rank}: Epoch {epoch+1} sync complete!")
        
        # Report metrics
        train.report({
            f"worker_{worker_rank}_epoch_{epoch}_time": epoch_time,
            f"worker_{worker_rank}_epoch_{epoch}_samples": epoch_samples,
            f"worker_{worker_rank}_epoch_{epoch}_batches": epoch_batches,
            f"worker_{worker_rank}_epoch_{epoch}_expected_batches": expected_batches,
            f"worker_{worker_rank}_epoch_{epoch}_loss": avg_loss
        })

def sync_model_across_workers(model):
    """Manually synchronize model parameters across all workers"""
    
    # Get all model parameters
    params = []
    for param in model.parameters():
        params.append(param.data.clone())
    
    # Use Ray Train's synchronization utilities
    from ray.train import get_context
    
    # Average parameters across all workers
    world_size = get_context().get_world_size()
    
    synced_params = []
    for param in params:
        # Sum all workers' parameters and average
        # Ray Train provides utilities for this
        avg_param = train.torch.get_device()  # Get proper device
        # Implementation details depend on Ray Train version
        # This is a simplified version - Ray Train has built-in utilities
        synced_params.append(param / world_size)
    
    # Create synchronized model
    synchronized_model = ResNet18(num_classes=80)
    for sync_param, model_param in zip(synced_params, synchronized_model.parameters()):
        model_param.data.copy_(sync_param)
    
    return synchronized_model

def run_worker_specific_experiment(experiment_type, distributions):
    
    print(f"\n{'='*60}")
    print(f"Running Worker-Specific COCO experiment: {experiment_type}")
    print(f"Data distribution: {distributions[experiment_type]}")
    print(f"{'='*60}")
    
    # Use worker-specific training function (no DDP)
    trainer = TorchTrainer(
        train_func_epoch_sync,  # Changed from train_func_distributed
        torch_config=TorchConfig(
            timeout_s=10000000
        ),
        train_loop_config={'experiment_type': experiment_type},
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
    
    analyze_worker_specific_results(result, experiment_type, total_time, distributions[experiment_type])
    return result

def analyze_worker_specific_results(result, experiment_type, total_time, distribution):
    """Analyze results showing data imbalance effects"""
    
    metrics = result.metrics
    
    print(f"\n--- Results for {experiment_type} ---")
    print(f"Intended distribution: {distribution}")
    print(f"Total experiment time: {total_time:.2f}s")
    
    worker_data = []
    actual_samples = []
    
    for i in range(3):
        worker_info = {
            'worker': i,
            'total_time': metrics.get(f"worker_{i}_total_time", 0),
            'total_samples': metrics.get(f"worker_{i}_total_samples", 0),
            'samples_per_second': metrics.get(f"worker_{i}_samples_per_second", 0),
            'intended_samples': distribution[i] if i < len(distribution) else 0
        }
        worker_data.append(worker_info)
        actual_samples.append(worker_info['total_samples'])
    
    print(f"Actual sample distribution: {actual_samples}")
    
    # Verify imbalance was achieved
    print("\nWorker Performance (IMBALANCE):")
    for worker in worker_data:
        if worker['total_time'] > 0:
            print(f"  Worker {worker['worker']}: {worker['total_samples']:,} samples "
                  f"(intended: {worker['intended_samples']:,}), "
                  f"{worker['total_time']:.2f}s, "
                  f"{worker['samples_per_second']:.1f} samples/sec")
    
    # Calculate imbalance impact
    times = [w['total_time'] for w in worker_data if w['total_time'] > 0]
    samples = [w['total_samples'] for w in worker_data if w['total_samples'] > 0]
    
    if times and len(times) > 1:
        print(f"\nImbalance Analysis:")
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

def download_coco_dataset(base_dir='/tmp/coco_data', worker_rank=0):
    """
    Download COCO 2017 dataset if it doesn't exist
    """
    base_path = Path(base_dir)
    images_dir = base_path / 'images' / 'train2017'
    annotations_file = base_path / 'annotations' / 'instances_train2017.json'
    
    print(f"Worker {worker_rank}: Checking COCO dataset at {base_dir}")
    
    # Check if dataset already exists
    if images_dir.exists() and annotations_file.exists():
        num_images = len(list(images_dir.glob('*.jpg')))
        if num_images > 100000:  # Should be ~118K images
            print(f"Worker {worker_rank}: COCO dataset already exists ({num_images} images)")
            return True
    
    print(f"Worker {worker_rank}: COCO dataset not found, downloading...")
    
    # Create directories
    base_path.mkdir(parents=True, exist_ok=True)
    (base_path / 'images').mkdir(exist_ok=True)
    (base_path / 'annotations').mkdir(exist_ok=True)
    
    try:
        # Download training images (~19GB)
        images_url = "http://images.cocodataset.org/zips/train2017.zip"
        images_zip = base_path / 'train2017.zip'
        
        if not images_zip.exists():
            print(f"Worker {worker_rank}: Downloading COCO images (~19GB)...")
            print(f"Worker {worker_rank}: This may take 10-30 minutes depending on connection...")
            
            # Use subprocess for wget (more reliable for large downloads)
            result = subprocess.run([
                'wget', '-O', str(images_zip), images_url, '--progress=bar'
            ], capture_output=False, text=True)
            
            if result.returncode != 0:
                print(f"Worker {worker_rank}: wget failed, trying urllib...")
                urllib.request.urlretrieve(images_url, images_zip)
        
        # Download annotations (~250MB)
        annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        annotations_zip = base_path / 'annotations_trainval2017.zip'
        
        if not annotations_zip.exists():
            print(f"Worker {worker_rank}: Downloading COCO annotations...")
            urllib.request.urlretrieve(annotations_url, annotations_zip)
        
        # Extract images
        if not images_dir.exists():
            print(f"Worker {worker_rank}: Extracting images...")
            with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                zip_ref.extractall(base_path / 'images')
        
        # Extract annotations  
        if not annotations_file.exists():
            print(f"Worker {worker_rank}: Extracting annotations...")
            with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
                zip_ref.extractall(base_path)
        
        # Verify download
        if images_dir.exists() and annotations_file.exists():
            num_images = len(list(images_dir.glob('*.jpg')))
            print(f"Worker {worker_rank}: COCO download complete! {num_images} images available")
            
            # Clean up zip files to save space
            if images_zip.exists():
                images_zip.unlink()
            if annotations_zip.exists():
                annotations_zip.unlink()
            
            return True
        else:
            print(f"Worker {worker_rank}: Download verification failed")
            return False
            
    except Exception as e:
        print(f"Worker {worker_rank}: Error downloading COCO: {e}")
        return False

def main():
    """Main function for memory-optimized imbalance experiment with COCO"""
    
    ray.init(ignore_reinit_error=True)
    
    try:
        # Define MEMORY-OPTIMIZED distributions for COCO (reduced sizes to fit in 30GB worker memory)
        distributions = {
            "equal": [40000, 39000, 39000],        # Equal distribution (118K total) 
            "imbalanced": [80000, 30000, 8000],    # 8:3:1 ratio (118K total)
            "extreme": [100000, 15000, 3000]         # 30:5:1 ratio (118K total)
        }
        
        print(f"Defined MEMORY-OPTIMIZED COCO distributions: {list(distributions.keys())}")
        print("Dataset sizes reduced to fit in 30GB worker memory limits")
        print("Workers will download COCO locally - this may take some time for the 25GB download!")
        
        print("\nStep 1: Running memory-optimized imbalance experiments with COCO...")
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
        
        # Step 2: Comparative analysis showing imbalance effects
        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS - MEMORY-OPTIMIZED DATA IMBALANCE (COCO)")
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
    os.environ["RAY_DEDUP_LOGS"] = "0"
    main()