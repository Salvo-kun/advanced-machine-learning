
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model.attention import LCAttention, LSAttention, GCAttention, GSAttention

def gem(x, p=torch.ones(1)*3, eps: float = 1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)

class GLAM(nn.Module):
    def __init__(self, features_dim_in, reduced_channels_dim, kernel_size=3, fusion_weights=None):
        super().__init__()
        self.local_channel_attention = LCAttention(kernel_size)
        self.global_channel_attention = GCAttention(kernel_size)
        self.local_spatial_attention = LSAttention(features_dim_in, reduced_channels_dim)
        self.global_spatial_attention = GSAttention(features_dim_in, reduced_channels_dim)

        self.fusion_weights = (
            Parameter(torch.Tensor([0.33, 0.33, 0.34])) if fusion_weights is None else fusion_weights
        )

    def forward(self, x):
        local_channel = self.local_channel_attention(x)
        global_channel = self.global_channel_attention(x)
        local_spatial = self.local_spatial_attention(x, local_channel)
        global_spatial = self.global_spatial_attention(x, global_channel)

        local_spatial = local_spatial.unsqueeze(1)
        global_spatial = global_spatial.unsqueeze(1)
        x = x.unsqueeze(1)

        concat = torch.cat((local_spatial, x, global_spatial), dim=1)
        weights = self.fusion_weights.softmax(-1).reshape(1, 3, 1, 1, 1)
        return (concat * weights).sum(1)