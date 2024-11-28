import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        norm_block,
        stride=2,
        padding=1,
        bias=False,
    ):
        super().__init__()
        if norm_block:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.LeakyReLU(),
            )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        final_block,
        dropout,
        stride=2,
        padding=1,
        bias=False,
    ):
        super().__init__()
        if final_block:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.Tanh(),
            )
        elif dropout:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.InstanceNorm2d(out_channels),
                nn.Dropout(0.5),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(),
            )

    def forward(self, x, skip_connection_x=None):
        if skip_connection_x is not None:
            return torch.cat([self.conv(x), skip_connection_x], axis=1)
        return self.conv(x)


class Generator(nn.Module):

    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64, kernel_size, False, 2, 1, bias=False)
        self.enc2 = EncoderBlock(64, 128, kernel_size, True, 2, 1, bias=False)
        self.enc3 = EncoderBlock(128, 256, kernel_size, True, 2, 1, bias=False)
        self.enc4 = EncoderBlock(256, 512, kernel_size, True, 2, 1, bias=False)
        self.enc5 = EncoderBlock(512, 512, kernel_size, True, 2, 1, bias=False)
        self.enc6 = EncoderBlock(512, 512, kernel_size, False, 2, 1, bias=False)

        self.dec1 = DecoderBlock(512, 512, kernel_size, False, True, 2, 1, bias=False)
        self.dec2 = DecoderBlock(1024, 512, kernel_size, False, True, 2, 1, bias=False)
        self.dec3 = DecoderBlock(1024, 256, kernel_size, False, True, 2, 1, bias=False)

        self.dec4 = DecoderBlock(512, 128, kernel_size, False, False, 2, 1, bias=False)
        self.dec5 = DecoderBlock(256, 64, kernel_size, False, False, 2, 1, bias=False)
        self.dec6 = DecoderBlock(128, 32, kernel_size, False, False, 2, 1, bias=False)

        self.dec7 = DecoderBlock(36, in_channels, (3, 3), True, False, 1, 1, bias=False)

    def forward(self, x):
        x0 = self.enc1(x)
        x1 = self.enc2(x0)
        x2 = self.enc3(x1)
        x3 = self.enc4(x2)
        x4 = self.enc5(x3)
        x5 = self.enc6(x4)

        x6 = self.dec1(x5, x4)
        x7 = self.dec2(x6, x3)
        x8 = self.dec3(x7, x2)
        x9 = self.dec4(x8, x1)
        x10 = self.dec5(x9, x0)
        self.x11 = self.dec6(x10, x)
        return self.dec7(self.x11)


class Discriminator(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.conv1 = EncoderBlock(
            in_channels * 2, 64, kernel_size, False, 2, 0, bias=False
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size, 1, "same", bias=False), nn.Sigmoid()
        )

    def forward(self, x, condition):
        x0 = self.conv1(torch.cat([x, condition], axis=1))
        return self.conv2(x0)
