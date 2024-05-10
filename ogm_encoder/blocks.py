import torch
import torch.nn.functional as F
from torch import nn

class ConvLayer(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(ConvLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size, stride, padding),
            nn.BatchNorm2d(n_out, eps=1e-05, momentum=0.05, affine=True),
            nn.Mish()
        )

    def forward(self, x):
        return self.seq(x)

class ResBlock(nn.Module):
    def __init__(self, n_in):
        super(ResBlock, self).__init__()
        self.branch = ConvLayer(n_in, n_in)

    def forward(self, x):
        return x + self.branch(x)
    
class ConvLayerTranspose(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=2, padding=0):
        super(ConvLayerTranspose, self).__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(n_in, n_out, kernel_size, stride, padding),
            nn.BatchNorm2d(n_out),
            nn.Mish()
        )

    def forward(self, x):
        return self.seq(x)