"""
    Parts of the U-Net model
    adapted from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

    I extended those functions to support time embedding.
    ref: https://nn.labml.ai/diffusion/ddpm/unet.html

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DoubleConvWithTimeEmbedding(nn.Module):
    """ (convolution => [BN] => Swish) => Time Embedding => (convolution => [BN] => Swish) """
    def __init__(self, channels_in, channels_out, channels_time):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            Swish()
        )
        self.time_emb = nn.Sequential(
            nn.Linear(channels_time, channels_out),
            Swish()
        )
        self.conv2 = nn.Sequential(    
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            Swish()
        )


    def forward(self, x, t_emb):
        x = self.conv1(x)
        x += self.time_emb(t_emb)[:, :, None, None]
        x = self.conv2(x)
        return x


class Down(nn.Module):
    """ Down-scaling with MaxPool and then DoubleConvWithTimeEmbedding() """
    def __init__(self, channels_in, channels_out, channels_time):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConvWithTimeEmbedding(channels_in, channels_out, channels_time)


    def forward(self, x, t_emb):
        x = self.maxpool(x)
        return self.conv(x, t_emb)


class Up(nn.Module):
    """ Up-scaling and then DoubleConvWithTimeEmbedding() """
    def __init__(self, channels_in, channels_out, channels_time):
        super().__init__()
        self.up = nn.ConvTranspose2d(channels_in, channels_in // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvWithTimeEmbedding(channels_in, channels_out, channels_time)

    def forward(self, x1, x2, t_emb):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, t_emb)


class Conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, padding=None):
        super().__init__()
        if padding:
            self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, padding=padding)
        else:
            self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)
