import tonic
import torch
from torch.utils.data import DataLoader
import numpy as np

# Bypassing the AWS WAF block by using direct figshare links
tonic.datasets.DVSGesture.train_url = "https://ndownloader.figshare.com/files/38022171"
tonic.datasets.DVSGesture.test_url = "https://ndownloader.figshare.com/files/38020584"

def get_collate_fn(device):
    def custom_collate(batch):
        """
        Collate function to format batches to [Time, Batch, Channels, Height, Width]
        and explicitly map tensors to the target device.
        """
        # DVSGesture time lengths can vary. For real-time simulation, we crop/pad to a fixed sequence length.
        # A time_window of 10,000us (10ms) implies 100 frames = 1 second.
        # We will fix T = 50 (500ms) for consistent tensor sizes.
        fixed_time = 50
        
        padded_frames = []
        targets = []
        for frames, target in batch:
            # frames shape: [T, C, H, W] from ToFrame
            tensor_frames = torch.from_numpy(frames).float()
            T_curr = tensor_frames.shape[0]
            
            if T_curr < fixed_time:
                pad = torch.zeros(fixed_time - T_curr, *tensor_frames.shape[1:])
                tensor_frames = torch.cat([tensor_frames, pad], dim=0)
            else:
                tensor_frames = tensor_frames[:fixed_time]
                
            padded_frames.append(tensor_frames)
            targets.append(target)
            
        # Stack along batch dimension: [Batch, Time, Channels, Height, Width]
        batch_frames = torch.stack(padded_frames)
        
        # Transpose to [Time, Batch, Channels, Height, Width] for SpikingJelly
        batch_frames = batch_frames.transpose(0, 1)
        batch_targets = torch.tensor(targets)
        
        return batch_frames.to(device), batch_targets.to(device)
    return custom_collate

def get_dataloader(batch_size=4, time_window=10000, train=True, device='cpu'):
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    
    transform = tonic.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_window)
    ])
    
    dataset = tonic.datasets.DVSGesture(save_to='./datasets', transform=transform, train=train)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=get_collate_fn(device),
        drop_last=True
    )
    return dataloader

if __name__ == "__main__":
    print("Initializing DataLoader and verifying tensor shapes...")
    loader = get_dataloader(batch_size=2)
    for x, y in loader:
        print(f"Batch shape (T, B, C, H, W): {x.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Device: {x.device}")
        break
