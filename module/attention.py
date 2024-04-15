import math
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):

    def __init__(self, channel):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        x = self.ca(x) * x
        x = self.sa(x) * x
        x += residual
        return x


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=128, w=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')

        weight = torch.view_as_complex(self.complex_weight)

        x = x * weight

        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')

        x = x.reshape(B, C, H, W)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, channel, image_size, is_out=False):
        super(AttentionBlock, self).__init__()
        self.channel = channel
        self.cbam = CBAM(channel)

        h_dict = dict([(64, image_size),
                       (128, image_size // 2),
                       (256, image_size // 4),
                       (512, image_size // 8),
                       (1024, image_size // 16)])

        h = h_dict[channel]

        if is_out:
            h = image_size
        w = h // 2 + 1

        self.filter = GlobalFilter(channel, h, w)

    def forward(self, x):
        y = self.cbam(x)
        z = self.filter(x)
        return y + z


if __name__ == '__main__':
    x = torch.rand(1, 64, 224, 224)
    # x = torch.rand(1, 256, 256, 64)
    model = AttentionBlock(64, 224)
    # model = GlobalFilter(64, h=256, w=256 // 2 + 1)

    out = model(x)

    print(out.shape)
