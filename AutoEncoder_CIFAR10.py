
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self._encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),       # 32x32 -> 16x16
            nn.ReLU(), 
            nn.Conv2d(16, 32, 3, stride=2, padding=1),      # 16x16 -> 8x8
            nn.ReLU(), 
            nn.Conv2d(32, 64, 3, stride=2, padding=1),      # 8x8 -> 4x4
            nn.ReLU()
        )
    
        self._fc1 = nn.Linear(64*4*4, 64)
        self._fc2 = nn.Linear(64, 64*4*4)

        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(), 
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
