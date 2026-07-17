import torch
import torch.nn as nn


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_ch, out_ch)
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

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, int_channels):
        super(AttentionBlock, self).__init__()
        self.Wx = nn.Sequential(nn.Conv3d(in_channels_x, int_channels, kernel_size=1),
                                nn.BatchNorm3d(int_channels))
        self.Wg = nn.Sequential(nn.Conv3d(in_channels_g, int_channels, kernel_size=1, bias=True),
                                nn.BatchNorm3d(int_channels))
        self.psi = nn.Sequential(nn.Conv3d(int_channels, 1, kernel_size=1, bias=True),
                                 nn.BatchNorm3d(1),
                                 nn.Sigmoid())

    def forward(self, x, g):
        # apply the Wx to the skip connection
        x1 = self.Wx(x)
        g1 = self.Wg(g)
        out = self.psi(nn.ReLU(inplace=True)(x1 + g1))
        return out * x


class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, out_channels):
        super(AttentionUpBlock, self).__init__()
        # self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionBlock(in_channels_x, in_channels_g, in_channels_g)
        self.conv_bn1 = DoubleConv(in_channels_g * 2, out_channels)
        self.conv_bn2 = DoubleConv(out_channels, out_channels)

    def forward(self, x, x_skip):
        # note : x_skip is the skip connection and x is the input from the previous block
        # apply the attention block to the skip connection, using x as context

        x = nn.functional.interpolate(x, x_skip.shape[2:], mode='trilinear', align_corners=False)
        x_attention = self.attention(x_skip, x)

        # stack their channels to feed to both convolution blocks
        x = torch.cat((x_attention, x), dim=1)
        x = self.conv_bn1(x)
        return self.conv_bn2(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        features = [32,64,128,256]

        self.inc = InConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[3])


        self.up1 = AttentionUpBlock(features[3], features[3], features[2])
        self.up2 = AttentionUpBlock(features[2], features[2], features[1])
        self.up3 = AttentionUpBlock(features[1], features[1], features[0])
        self.up4 = AttentionUpBlock(features[0], features[0], features[0])
        self.outc = OutConv(features[0], num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 2, 64, 64, 64)
    net = UNet(in_channels=2, num_classes=4)
    y = net(x)
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)