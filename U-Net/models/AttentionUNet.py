import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttnUNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.attention_blocks = nn.ModuleList()

        # Down
        in_channels = config.MODEL.IN_CHANNELS
        for feature in config.MODEL.FEATURES:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Attention
        for feature in config.MODEL.FEATURES:
            self.attention_blocks.append(
                AttentionBlock(F_g=feature, F_l=feature, F_int=feature // 2)
            )

        # Up
        for feature in reversed(config.MODEL.FEATURES):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Finale Conv
        self.final_conv = nn.Conv2d(
            config.MODEL.FEATURES[0],
            config.MODEL.OUT_CHANNELS,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        for i, up in enumerate(self.ups):
            x = up(x)
            skip = skip_connections[-(i + 2)]
            x = self.attention_blocks[i](g=x, x=skip)
            x = torch.cat((skip, x), dim=1)

        x = self.final_conv(x)
        return x
