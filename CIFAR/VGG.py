import torch
import torch.nn as nn

import TCSAttention
from spikingjelly.clock_driven import surrogate, layer, neuron, functional
import transforms


class VGG_CIFAR(nn.Module):  # test origin modlue
    def __init__(self, T: int = 14, kernel_size=3, rank=64, device='cpu'):
        super().__init__()
        conv = []
        conv.append(layer.SeqToANNContainer(nn.AdaptiveAvgPool2d(48)))
        conv.append(
            TCSAttention.TCSAModule(H=48, W=48, T=T, C=2, kernel_sizeTS=kernel_size, rank=rank, device=device))
        conv.extend(VGG_CIFAR.conv3x3(2, 64))
        conv.extend(VGG_CIFAR.conv3x3(64, 128))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            TCSAttention.TCSAModule(H=24, W=24, T=T, C=128, kernel_sizeTS=kernel_size, rank=rank, device=device))

        conv.extend(VGG_CIFAR.conv3x3(128, 256))
        conv.extend(VGG_CIFAR.conv3x3(256, 256))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            TCSAttention.TCSAModule(H=12, W=12, T=T, C=256, kernel_sizeTS=kernel_size, rank=rank, device=device))

        conv.extend(VGG_CIFAR.conv3x3(256, 512))
        conv.extend(VGG_CIFAR.conv3x3(512, 512))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            TCSAttention.TCSAModule(H=6, W=6, T=T, C=512, kernel_sizeTS=kernel_size, rank=rank, device=device))

        conv.extend(VGG_CIFAR.conv3x3(512, 512))
        conv.extend(VGG_CIFAR.conv3x3(512, 512))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            TCSAttention.TCSAModule(H=3, W=3, T=T, C=512, kernel_sizeTS=kernel_size, rank=rank, device=device))

        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            nn.Flatten(2),
            layer.SeqToANNContainer(nn.Linear(512 * 3 * 3, 10)),
            neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        )

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        out_spikes = self.fc(self.conv(x))  # shape = [T, N, 101]
        return out_spikes.mean(0)

    @staticmethod
    def conv3x3(in_channels: int, out_channels):
        return [
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy'),
        ]


class VGG_NCAL(nn.Module):  # test origin modlue
    def __init__(self, T: int = 14, kernel_size=3, rank=64, device='cpu'):
        super().__init__()
        conv = []
        conv.append(layer.SeqToANNContainer(nn.AdaptiveAvgPool2d(48)))
        conv.append(
            TCSAttention.TCSAModule(H=48, W=48, T=T, C=2, kernel_sizeTS=kernel_size, rank=rank, device=device))
        conv.extend(VGG_NCAL.conv3x3(2, 64))
        conv.extend(VGG_NCAL.conv3x3(64, 128))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            TCSAttention.TCSAModule(H=24, W=24, T=T, C=128, kernel_sizeTS=kernel_size, rank=rank, device=device))

        conv.extend(VGG_NCAL.conv3x3(128, 256))
        conv.extend(VGG_NCAL.conv3x3(256, 256))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            TCSAttention.TCSAModule(H=12, W=12, T=T, C=256, kernel_sizeTS=kernel_size, rank=rank, device=device))

        conv.extend(VGG_NCAL.conv3x3(256, 512))
        conv.extend(VGG_NCAL.conv3x3(512, 512))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            TCSAttention.TCSAModule(H=6, W=6, T=T, C=512, kernel_sizeTS=kernel_size, rank=rank, device=device))

        conv.extend(VGG_NCAL.conv3x3(512, 512))
        conv.extend(VGG_NCAL.conv3x3(512, 512))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(
            TCSAttention.TCSAModule(H=3, W=3, T=T, C=512, kernel_sizeTS=kernel_size, rank=rank, device=device))

        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            nn.Flatten(2),
            layer.SeqToANNContainer(nn.Linear(512 * 3 * 3, 101)),
            neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        )

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        out_spikes = self.fc(self.conv(x))  # shape = [T, N, 101]
        return out_spikes.mean(0)

    @staticmethod
    def conv3x3(in_channels: int, out_channels):
        return [
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy'),
        ]


class VGG_Gesture(nn.Module):  # test origin modlue
    def __init__(self, T: int = 20, kernel_size=3, rank=32, device='cuda:0'):
        super().__init__()
        conv = []
        conv.append(TCSAttention.TCSAModule(128, 128, T, 2, rank=rank,device=device))
        conv.extend(VGG_NCAL.conv3x3(2, 64))
        conv.extend(VGG_NCAL.conv3x3(64, 128))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(TCSAttention.TCSAModule(64, 64, T, 128, rank=rank,device=device))
        conv.extend(VGG_NCAL.conv3x3(128, 256))
        conv.extend(VGG_NCAL.conv3x3(256, 256))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(TCSAttention.TCSAModule(32, 32, T, 256, rank=rank, device=device))
        conv.extend(VGG_NCAL.conv3x3(256, 512))
        conv.extend(VGG_NCAL.conv3x3(512, 512))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(TCSAttention.TCSAModule(16, 16, T, 512, rank=rank, device=device))

        conv.extend(VGG_NCAL.conv3x3(512, 512))
        conv.extend(VGG_NCAL.conv3x3(512, 512))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        conv.append(TCSAttention.TCSAModule(8, 8, T, 512, device=device))
        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            nn.Flatten(2),
            layer.SeqToANNContainer(nn.Linear(512 * 8 * 8, 110)),
            neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        )

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        out_spikes = self.fc(self.conv(x))  # shape = [T, N, 101]
        return out_spikes.mean(0)

    @staticmethod
    def conv3x3(in_channels: int, out_channels):
        return [
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            neuron.MultiStepLIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy'),
        ]

