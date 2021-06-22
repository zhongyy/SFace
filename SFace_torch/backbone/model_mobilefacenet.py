from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import OrderedDict
import math

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Conv_block(Module): # verified: the same as ``Conv'' in ./fmobilefacenet
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv2d = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False) # verified: the same as mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True)
        self.batchnorm = BatchNorm2d(out_c, eps=0.001) # verified: the same as mx.sym.BatchNorm(data=conv, fix_gamma=False,momentum=0.9)
        self.relu = PReLU(out_c)
    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class Linear_block(Module): # verified: the same as ``Linear'' in ./fmobilefacenet
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv2d = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.batchnorm = BatchNorm2d(out_c, eps=0.001)
    def forward(self, x):
        x = self.conv2d(x)
        x = self.batchnorm(x)
        return x

class Depth_Wise(Module): # verified: if residual is False: the same as ``DResidual'' in ./fmobilefacenet; else: the same as ``Residual''
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv_sep = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.conv_proj = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv_sep(x)
        x = self.conv_dw(x)
        x = self.conv_proj(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module): # verified: the same as ``Residual'' in ./fmobilefacenet
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = OrderedDict()
        for i in range(num_block):
            modules['block%d'%i] = Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups)
        self.model = Sequential(modules)
    def forward(self, x):
        return self.model(x)

class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.conv_1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.dconv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.res_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dconv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.res_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dconv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.res_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6dw7_7 = Linear_block(512, 512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.pre_fc1 = Linear(512, embedding_size)
        self.fc1 = BatchNorm1d(embedding_size, eps=2e-5) # doubt: the same as mx.sym.BatchNorm(data=conv_6_f, fix_gamma=True, eps=2e-5, momentum=0.9)?
    
    def forward(self, x):
        x = x - 127.5
        x = x*0.078125
        out = self.conv_1(x)
        out = self.conv_2_dw(out)
        out = self.dconv_23(out)
        out = self.res_3(out)
        out = self.dconv_34(out)
        out = self.res_4(out)
        out = self.dconv_45(out)
        out = self.res_5(out)
        out = self.conv_6sep(out)
        out = self.conv_6dw7_7(out)
        out = self.conv_6_flatten(out)
        out = self.pre_fc1(out)
        out = self.fc1(out)
        return out
