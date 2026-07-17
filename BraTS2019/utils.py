import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from medpy import metric

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def Dice(output, target, eps=1e-3):
    inter = torch.sum(output * target,dim=(1,2,-1)) + eps
    union = torch.sum(output,dim=(1,2,-1)) + torch.sum(target,dim=(1,2,-1)) + eps * 2
    x = 2 * inter / union
    dice = torch.mean(x)
    return dice


def cal_dice(output, target):
    output = torch.argmax(output,dim=1)
    dice1 = Dice((output == 3).float(), (target == 3).float())
    dice2 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    dice3 = Dice((output != 0).float(), (target != 0).float())

    return dice1, dice2, dice3

class Loss(nn.Module):
    def __init__(self, n_classes, weight=None, alpha=0.5):  
        "dice_loss_plus_cetr_weighted"  
        super(Loss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight.cuda()
        self.alpha = alpha

    def forward(self, input, target):
        smooth = 0.01

        input1 = F.softmax(input, dim=1)
        target1 = F.one_hot(target, self.n_classes)
        input1 = rearrange(input1, 'b n h w s -> b n (h w s)')
        target1 = rearrange(target1, 'b h w s n -> b n (h w s)')


        inter1 = torch.sum(input1[:,1,:] * target1[:,1,:])
        union1 = torch.sum(input1[:,1,:]) + torch.sum(target1[:,1,:]) + smooth
        L_dice1 = 1.0 - 2.0 * inter1 / union1
        
        inter2 = torch.sum(input1[:,2,:] * target1[:,2,:])
        union2 = torch.sum(input1[:,2,:]) + torch.sum(target1[:,2,:]) + smooth
        L_dice2 = 1.0 - 2.0 * inter2 / union2

        inter3 = torch.sum(input1[:,3,:] * target1[:,3,:])
        union3 = torch.sum(input1[:,3,:]) + torch.sum(target1[:,3,:]) + smooth
        L_dice3 = 1.0 - 2.0 * inter3 / union3

        L_ce = F.cross_entropy(input,target, weight=self.weight)
        L_dice = L_dice1 + L_dice2 + L_dice3
        # print((1 - self.alpha) * loss, (1 - dice) * self.alpha)
        return L_ce, L_dice


if __name__ == '__main__':
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losser = Loss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).to(device)
    x = torch.randn((2, 4, 16, 16, 16)).to(device)
    y = torch.randint(0, 4, (2, 16, 16, 16)).to(device)
    print(losser(x, y))
    print(cal_dice(x, y))
