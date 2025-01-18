import torch
from torch import nn

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = normalized_shape

        self.dynamic_shape = None
    
    def forward(self, x: torch.Tensor, dim: int=1) -> torch.Tensor:
        u = x.mean(dim, keepdim=True)
        s = (x - u).pow(2).mean(dim, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)

        if self.dynamic_shape is None:
            new_shape = torch.tensor(list(x.shape), dtype=torch.int16) == self.normalized_shape
            self.dynamic_shape = torch.where(new_shape, self.normalized_shape, 1).tolist()
            
        x = self.weight.reshape((-1, 1, 1, 1)) * x + self.bias.reshape((-1, 1, 1, 1))
        return x
        
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x