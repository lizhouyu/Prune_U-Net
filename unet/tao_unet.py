import torch
import torch.nn as nn

from .tao_unet_blocks import ResDownBlock, ResUpBlock

class UNet(nn.Module
):
    def __init__(self,
                 in_channels: int = 3, 
                 out_channels: int = 1, 
                ) -> None:
        super(UNet, self).__init__()

        layer_dim_list = [64, 128, 256, 512]

        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels, layer_dim_list[0], kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=False)
        )

        self.down1 = ResDownBlock(layer_dim_list[0], layer_dim_list[0]) # dim: 64 -> 64
        self.down2 = ResDownBlock(layer_dim_list[0], layer_dim_list[1]) # dim: 64 -> 128
        self.down3 = ResDownBlock(layer_dim_list[1], layer_dim_list[2]) # dim: 128 -> 256
        self.down4 = ResDownBlock(layer_dim_list[2], layer_dim_list[3], conv_stride = 1) # dim: 256 -> 512
        

        self.up4 = nn.ConvTranspose2d(layer_dim_list[3], layer_dim_list[2], kernel_size=4, stride=2, padding=1) # dim: 512 -> 256
        self.up3 = ResUpBlock(layer_dim_list[2] + layer_dim_list[1], layer_dim_list[1])  # dim: 256 + 128 -> 128
        self.up2 = ResUpBlock(layer_dim_list[1] + layer_dim_list[0], layer_dim_list[0])  # dim: 128 + 64 -> 64
        self.up1 = ResUpBlock(layer_dim_list[0] + layer_dim_list[0], layer_dim_list[0]) # dim: 64 + 64 -> 64

        self.outconv = nn.Sequential(
            nn.Conv2d(layer_dim_list[0], layer_dim_list[0], kernel_size=3, padding=1), # dim: 64 -> 64
            nn.ReLU(inplace=False),
            nn.Conv2d(layer_dim_list[0], out_channels, kernel_size=3, padding=1), # dim: 64 -> 1
            nn.ReLU(inplace=False)
        )

        self.activation = nn.ReLU(inplace=False)

        self.Sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.inconv(x)
        x = self.activation(x) # dim: in -> 64
        # print("x", x.shape)

        down1 = self.down1(x) # dim: 64 -> 64
        # print("down1", down1.shape)
        down2 = self.down2(down1) # dim: 64 -> 128
        # print("down2", down2.shape)
        down3 = self.down3(down2) # dim: 128 -> 256
        # print("down3", down3.shape)
        down4 = self.down4(down3) # dim: 256 -> 512
        # print("down4", down4.shape)

        up4 = self.up4(down4) # dim: 512 -> 256
        # print("up4", up4.shape)
        up4 = torch.cat([down2, up4], dim=1) # dim: 256 + 128 = 384
        up4 = self.activation(up4)
        up3 = self.up3(up4) # dim: 384 -> 128
        # print("up3", up3.shape)
        up3 = torch.cat([down1, up3], dim=1) # dim: 128 + 64 = 192
        up3 = self.activation(up3)
        up2 = self.up2(up3) # dim: 192 -> 64
        # print("up2", up2.shape)
        up2 = torch.cat([x, up2], dim=1) # dim: 64 + 64 = 128
        up2 = self.activation(up2) 
        up1 = self.up1(up2) # dim: 128 -> 64
        # print("up1", up1.shape)

        out = self.outconv(up1) # dim: 64 -> out_channels
        out = self.Sigmoid(out)
        return out


if __name__ == '__main__':
    model = UNet(out_channels=2)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(y.shape)