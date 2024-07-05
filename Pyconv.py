import torch
import torch.nn as nn
#金字塔卷积
class PyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                               stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                               dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))
        return torch.cat(out, 1)

if __name__ == '__main__':
    #pyconv有2种levels  kernels： 3*3， 5*5
    m = PyConv2d(in_channels=64, out_channels=[32,32], pyconv_kernels=[3, 5], pyconv_groups=[1, 4])
    input = torch.rand(4,256,64)
    H = 16
    W = 16
    input_bchw = input.view(4, 64, H, W)
    output = m(input_bchw)
    print(output.shape)
    #pyconv有三种levels  kernels： 3*3， 5*5， 7*7
    m = PyConv2d(in_channels=2048, out_channels=[512,512,1024], pyconv_kernels=[3,5,7], pyconv_groups=[1,4,8])
    input = torch.rand(4,2048,56,56)
    output = m(input)
    print(output.shape)