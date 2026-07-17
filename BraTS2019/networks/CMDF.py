import torch
import torch.nn as nn
from networks.SimAM import SimAM
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CMDF(nn.Module):
    def __init__(self, channels):
        super(CMDF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(channels),
            nn.Sigmoid()
        )
        self.conv = nn.Conv3d(2 * channels, channels, kernel_size=3, stride=1, padding=1)
        self.simam = SimAM()
    def forward(self, x1, x2):
        x1 = self.simam(x1)
        x2 = self.simam(x2)
        dif = x1 - x2
        w_dif = self.conv1(dif)
        x1 = torch.cat([x1, x1 + w_dif * x1], dim=1)
        x2 = torch.cat([x2, x2 + w_dif * x2], dim=1)
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1,8,128,128,128).to(device)
    Model = CMDF(channels=8).cuda()

    out = Model(x) # (B,C,H,W)
    print(out.shape)
    print("params: ", sum(p.numel() for p in Model.parameters()))