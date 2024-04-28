import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops



class TCSAModule(nn.Module):
    def __init__(self, H: int = 32, W: int = 32, T: int = 12, C: int = 128, kernel_sizeTS: int = 3, rank: int = 64,
                 device='cpu'):
        super(TCSAModule, self).__init__()
        self.result = None
        self.SC_conv = None
        self.W = W
        self.H = H
        self.N = None
        self.rank = rank
        self.T = T
        self.C = C
        self.device = device

        self.TC_pooling = nn.AdaptiveAvgPool2d(1)
        self.TS_conv = nn.Conv2d(in_channels=self.T, out_channels=self.rank, kernel_size=kernel_sizeTS, stride=1,
                                 padding=1 if kernel_sizeTS == 3 else 0,
                                 bias=True)
        # self.TC_FC = nn.Sequential(*[nn.Linear(self.C, 4*self.C), nn.Sigmoid(), nn.Linear(4*self.C, 4*self.rank), nn.Sigmoid(), nn.Linear(4*self.rank, self.rank)])
        self.TC_FC = nn.Linear(self.C, self.rank)
        self.SC_FC = nn.Linear(self.T, self.rank)
        # self.SC_FC = nn.Sequential(*[nn.Linear(self.T, 4*self.T), nn.Sigmoid(), nn.Linear(4*self.T, 4*self.rank), nn.Sigmoid(), nn.Linear(4*self.rank, self.rank)])

    def forward(self, x):
        # x: TxNxCxHxW
        x = einops.rearrange(x, "n c h w t -> n t c h w")
        # x = x.permute(1, 0, 2, 3, 4)  # NxTxCxHxW
        [self.N, self.T, self.C, self.H, self.W] = x.size()
        # print("_________________________________INFO___________________________________________")
        # print(x.size())
        # print("________________________________________________________________________________")

        self.result = torch.zeros((self.N, self.T, self.C, self.H, self.W), requires_grad=False).to(self.device)
        # 我的想法是，直接把HW拉成一条，空间信息不就寄了吗，所以分门别类来用不同方法处理
        TS = x.permute(0, 1, 3, 4, 2)  # reshape x to NxTxHxWxC
        TS = torch.mean(TS, dim=4)
        TS = self.TS_conv(TS)  # NxRxHxW
        TS = TS.view((self.N, self.rank, self.H * self.W))
        TS = torch.sigmoid(TS).permute(1, 0, 2)  # RxNxHW


        TC = self.TC_pooling(x).view(self.N, self.T, self.C)  # NxTxC
        TC = self.TC_FC(TC).permute(0, 2, 1)
        TC = torch.sigmoid(TC).permute(1, 0, 2)  # RxNxT




        SC = self.TC_pooling(x).view(self.N, self.T, self.C).permute(0, 2, 1)  # NxCxT
        SC = self.SC_FC(SC)  # NxCxR
        SC = torch.sigmoid(SC).permute(2, 0, 1)  # RxNxC



        # 或许需要像原文一样，每一条都是一样的，然后1x1卷积，再sigmoid，先试试我的想法吧
        for i in range(0, self.rank):
            self.result += self.reconstruct(TS[i], TC[i], SC[i])
        # res = F.relu_(self.result * x).permute(1, 0, 2, 3, 4) if global_vars.epoch > global_vars.ablation_epoch else x.permute(1, 0, 2, 3, 4)
        res = F.relu_(self.result * x)

        return einops.rearrange(res, "n t c h w -> n c h w t")
        # return res*(torch.max(x)-torch.min(x))/(torch.max(res)-torch.min(res))

    def reconstruct(self, feat1: torch.Tensor, feat2: torch.Tensor, feat3: torch.Tensor):
        # feat1:NxHW feat2:NxT feat3:NxC
        S = feat1.view(self.N, -1, 1)  # NxHWx1
        T = feat2.view(self.N, 1, -1)  # Nx1xT
        C = feat3.view(self.N, 1, -1)  # Nx1xC
        # S bmm T = NxHWTx1
        # ST bmm x C = NxHWTxC
        return torch.bmm(torch.bmm(S, T).view(self.N, -1, 1), C).view(self.N, self.H, self.W, self.T, self.C) \
            .permute(0, 3, 4, 1, 2)

