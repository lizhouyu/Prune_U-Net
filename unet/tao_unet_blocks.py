import torch
import torch.nn as nn

class ResDownSubBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_stride: int = 1):
        super(ResDownSubBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=conv_stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=conv_stride, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_stride: int = 2): # stride=2 for downsampling, stride=1 for keeping the same size
        super(ResDownBlock, self).__init__()
        self.res_down_sub_block1 = ResDownSubBlock(in_channels, out_channels, conv_stride=conv_stride) # stride=2 for downsampling
        self.res_down_sub_block2 = ResDownSubBlock(out_channels, out_channels)

    def forward(self, x):
        out = self.res_down_sub_block1(x)
        out = self.res_down_sub_block2(out)
        return out

class ResUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResUpBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.convTranspose = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.convTranspose(out)
        out = self.bn2(out)
        return out



