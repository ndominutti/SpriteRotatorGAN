import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        first_block,
        stride=2,
        padding=1,
        bias=False,
    ):
        if first_block:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.functional.leaky_relu(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.InstanceNorm2d(out_channels),
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
        stride=2,
        padding=1,
        bias=False,
    ):
        if final_block:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.InstanceNorm2d(out_channels),
                nn.Tanh(),
            )

        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.InstanceNorm2d(out_channels),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

    def forward(self, x, skip_connection_x):
        x = torch.cat([x, skip_connection_x], axis=1)
        return self.conv(x)


class UNET(nn.Module):

    def __init__(self, in_channels, kernel_size):
        self.enc1 = EncoderBlock(in_channels, 64, kernel_size, True, 2, 1, bias=False)
        self.enc2 = EncoderBlock(64, 128, kernel_size, False, 2, 1, bias=False)
        self.enc3 = EncoderBlock(128, 256, kernel_size, False, 2, 1, bias=False)
        self.enc4 = EncoderBlock(256, 512, kernel_size, False, 2, 1, bias=False)
        self.enc5 = EncoderBlock(512, 512, kernel_size, False, 2, 1, bias=False)
        self.enc6 = EncoderBlock(512, 512, kernel_size, False, 2, 1, bias=False)

        self.dec1 = DecoderBlock(1024, 512, kernel_size, False, 2, 1, bias=False)
        self.dec2 = DecoderBlock(1024, 256, kernel_size, False, 2, 1, bias=False)
        self.dec3 = DecoderBlock(512, 128, kernel_size, False, 2, 1, bias=False)

        self.dec4 = DecoderBlock(256, 64, kernel_size, False, 2, 1, bias=False)
        self.dec5 = DecoderBlock(128, 32, kernel_size, False, 2, 1, bias=False)
        self.dec6 = DecoderBlock(36, in_channels, kernel_size, True, 2, 1, bias=False)

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
        return self.dec6(x10, x)
