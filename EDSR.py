import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        # out = out * 0.1  # EDSR의 residual scaling
        out = out + residual
        return out

class EDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, scale=4):
        super(EDSR, self).__init__()

        # 초기 특징 추출
        self.head = nn.Conv2d(3, n_feats, kernel_size=3, padding=1)

        # Residual Blocks
        self.body = nn.Sequential(
            *[ResBlock(n_feats) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        )

        # Upscaling
        self.upscale = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.upscale(res)
        return x