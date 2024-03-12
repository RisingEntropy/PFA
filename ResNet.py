import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer, neuron
from spikingjelly.clock_driven.surrogate import heaviside, SurrogateFunctionBase


class ms_surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.require_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):  # 源代码中alpha其实是1，不知道原作者在写什么

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = (1 / ctx.alpha) * torch.sign((ctx.saved_tensors[0] <= ctx.alpha / 2.).to(ctx.saved_tensors[0]))
        return grad_x, None


class MS(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return ms_surrogate.apply(x, alpha)

    @staticmethod
    def primitive_function(x, alpha):
        raise NotImplemented("No primitive function for MS")


class ResBlockRM3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockRM3, self).__init__()
        resdual = []
        resdual.append(neuron.MultiStepLIFNode(tau=0.25, v_threshold=0.5, surrogate_function=MS()))
        resdual.append(layer.SeqToANNContainer(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ))
        resdual.append(neuron.MultiStepLIFNode(tau=0.25, v_threshold=0.5, surrogate_function=MS()))
        resdual.append(layer.SeqToANNContainer(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ))
        self.resdual_part = nn.Sequential(*resdual)
        self.shortcut = nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.shortcut = layer.SeqToANNContainer(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                                              kernel_size=3,stride=stride,bias=False),
                                                    nn.BatchNorm2d(out_channels))
    def forward(self,x):
        return self.resdual_part(x)+x


class SMResNet_104(nn.Module):
    def __init__(self,num_block,num_classes=1000):
        super().__init__()
        self.conv1 = layer.SeqToANNContainer(
            nn.Conv2d(2,64,kernel_size=3,padding=1,)
        )