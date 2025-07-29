
from typing import Tuple

import torch
from torch import nn


class Encoder(nn.Module):
    '''
    Auto Encorderのエンコーダ
    '''

    def __init__(self):
        super().__init__()

        self._encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(16, 32, 3, stride=2, padding=1), 
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._encoder.forward(x)

class Decoder(nn.Module):
    '''
    Auto Encorderのデコーダ
    '''

    def __init__(self):
        super().__init__()

        self._decorder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(), 
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._decorder.forward(x)

class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self._encorder = Encoder()
        self._fc1 = nn.Linear(32*7*7, 64)
        self._fc2 = nn.Linear(64, 32*7*7)
        self._decorder =Decoder()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._encorder.forward(x)
        x = x.view(x.size(0), -1)
        x = self._fc1.forward(x)
        x = self._fc2.forward(x)
        x = x.view(x.size(0), 32, 7, 7)
        x = self._decorder.forward(x)
        return x
