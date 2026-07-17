import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class SimAM(torch.nn.Module):

    def __init__(self, epsilon=1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.epsilon = epsilon

    def forward(self, x):

        # (B,C,H,W)
        batch_size, channels, height, width, depth = x.size()

        n = width * height * depth - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)

        attention = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.epsilon)) + 0.5

        return x * self.activation(attention)



if __name__ == '__main__':
    # (B,C,H,W)
    x1 = torch.randn(1, 64, 128, 128, 128)
    B, C, H, W, D = x1.size()

    Model = SimAM()

    out = Model(x1)
    print(out.shape)

