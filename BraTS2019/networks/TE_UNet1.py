import torch
import torch.nn as nn

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = Conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2, 2),
            Conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        # x = self.sigmoid(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = Conv(in_ch+skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class TE_UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(TE_UNet, self).__init__()
        features = [32, 64, 128, 256]

        self.inc = InConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
    
        self.conv = Conv(features[3], features[3])

        self.up1 = Up(features[3], features[2], features[2])
        self.up2 = Up(features[2], features[1], features[1])
        self.up3 = Up(features[1], features[0], features[0])
        self.outc = OutConv(features[0], num_classes)

    def forward(self, x):
        x1 = self.inc(x)  # 2 -> 32

        x2 = self.down1(x1)  # 32 -> 64

        x3 = self.down2(x2)  # 64 -> 128

        x4 = self.down3(x3)  # 128 -> 256

        x = self.conv(x4)

        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x


if __name__ == '__main__':
    x1 = torch.randn(1, 4, 64, 64, 64)
    net = TE_UNet(in_channels=4, num_classes=4)
    y = net(x1)
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)