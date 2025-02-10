import torch
from torch import nn

class Stem(nn.Module):
    def __init__(self, resolution:int, in_channels: int, out_channels: int, reduction:int=1) -> None:
        super(Stem, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
        self.resolution = resolution

        self.stride1 = 2 if reduction % 2 == 0 else 1
        self.stride2 = 2 if reduction % 4 == 0 else 1

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride1, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=self.stride2, padding=2, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )

    def get_output_resolution(self) -> int:
        return (((self.resolution - 1) // self.stride1) + 1) // self.stride2 + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return x


class TakuBlock(nn.Module):
    def __init__(self, resolution: int, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int) -> None:
        super(TakuBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.resolution = resolution

        self.skip_conn = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else nn.Identity()

        self.bn = nn.BatchNorm2d(in_channels)
        self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=out_channels if in_channels==out_channels else 1)
        self.activation = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_conn(x)
        x = self.bn(self.dwconv(x))
        x = self.activation(x) + skip

        return x

