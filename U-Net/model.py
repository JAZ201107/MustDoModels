import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
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


class UNET(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down
        in_channels = config.MODEL.IN_CHANNELS
        for feature in config.MODEL.FEATURES:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up
        for feature in reversed(config.MODEL.FEATURES):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Bottleneck
        self.bottleneck = DoubleConv(
            config.MODEL.FEATURES[-1], config.MODEL.FEATURES[-1] * 2
        )

        # final conv
        self.final_conv = nn.Conv2d(
            config.MODEL.FEATURES[0], config.MODEL.OUT_CHANNELS, kernel_size=1
        )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            print("x.shape:", x.shape)
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            print("=> x.shape:", x.shape)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse the list

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 572, 572))
    config = get_config()
    config.MODEL.IN_CHANNELS = 1
    model = UNET(config)
    preds = model(x)
    assert x.shape == preds.shape
    print("Success")


if __name__ == "__main__":
    test()
