import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

class ShallowSNN(nn.Module):
    def __init__(self, in_channels=2, height=128, width=128, hidden_size=256, num_classes=11):
        super().__init__()
        self.flat_size = in_channels * height * width
        self.hidden_size = hidden_size
        
        # We use step_mode='s' (single-step) to allow explicit time-step iteration
        # in the forward pass. This enables mathematical lateral inhibition
        # strictly at the tensor level immediately when a neuron fires.
        # Spatial matrix operations remain fully vectorized on CUDA.
        
        self.fc1 = layer.Linear(self.flat_size, hidden_size, step_mode='s')
        self.lif1 = neuron.LIFNode(tau=2.0, step_mode='s', v_threshold=1.0)
        
        self.fc2 = layer.Linear(hidden_size, num_classes, step_mode='s')
        self.lif2 = neuron.LIFNode(tau=2.0, step_mode='s', v_threshold=1.0)
        
        # Lateral inhibition matrix for the hidden layer [Hidden_Size, Hidden_Size]
        # Diagonal is 0 (no self-inhibition), off-diagonal can be positive to represent
        # the amount of inhibition (it will be subtracted).
        lat_w = torch.ones(hidden_size, hidden_size) * 0.1
        lat_w.fill_diagonal_(0.0)
        self.lateral_weight = nn.Parameter(lat_w, requires_grad=False)
        
    def forward(self, x_seq, apply_inhibition=False):
        """
        x_seq: [Time, Batch, Channels, Height, Width]
        apply_inhibition: Toggle for lateral inhibition during learning
        """
        T, B, C, H, W = x_seq.shape
        # Flatten spatial dims to [Time, Batch, Flat_Size]
        x_seq_flat = x_seq.view(T, B, -1)
        
        s1_seq = []
        s2_seq = []
        
        # Reset internal states before processing a new batch
        functional.reset_net(self)
        
        for t in range(T):
            x_t = x_seq_flat[t]
            
            # Forward pass through first Linear layer (CUDA matrix op)
            current_input = self.fc1(x_t)
            
            # Apply Lateral Inhibition if enabled and past t=0
            if apply_inhibition and t > 0:
                # s1_seq[-1] contains spikes from previous timestep [Batch, Hidden_Size]
                # Matrix multiply yields inhibition tensor [Batch, Hidden_Size]
                inhibition = torch.matmul(s1_seq[-1], self.lateral_weight)
                
                # Apply negative bias directly to membrane potentials
                # to prevent winner-take-all weight collapse.
                self.lif1.v = self.lif1.v - inhibition
                
            # Fire LIF node
            s1 = self.lif1(current_input)
            s1_seq.append(s1)
            
            # Output layer
            x2 = self.fc2(s1)
            s2 = self.lif2(x2)
            s2_seq.append(s2)
            
        return torch.stack(s1_seq), torch.stack(s2_seq)

    def reset_states(self):
        functional.reset_net(self)

if __name__ == "__main__":
    print("Testing SNN forward pass...")
    # Mock data: Time=50, Batch=2, C=2, H=128, W=128
    mock_input = torch.rand(50, 2, 2, 128, 128).to('cuda')
    model = ShallowSNN().to('cuda')
    s1, s2 = model(mock_input, apply_inhibition=True)
    print(f"Hidden spikes shape: {s1.shape}")
    print(f"Output spikes shape: {s2.shape}")
