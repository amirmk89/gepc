import torch.nn as nn
from models.graph.pygeoconv import PyGeoConv


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1,
                 dropout=0,
                 conv_oper='sagc',
                 act=None,
                 out_bn=True,
                 out_act=True,
                 residual=True,
                 headless=False,):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.headless = headless
        self.out_act = out_act
        self.conv_oper = conv_oper
        self.act = nn.ReLU(inplace=True) if act is None else act
        self.gcn = PyGeoConv(in_channels, out_channels, kernel_size=kernel_size[1], dropout=dropout,
                             headless=self.headless, conv_oper=self.conv_oper)

        if out_bn:
            bn_layer = nn.BatchNorm2d(out_channels)
        else:
            bn_layer = nn.Identity()  # Identity layer shall no BN be used

        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 self.act,
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (kernel_size[0], 1),
                                           (stride, 1),
                                           padding),
                                 bn_layer,
                                 nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                          out_channels,
                                          kernel_size=1,
                                          stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x, adj):
        res = self.residual(x)
        x, adj = self.gcn(x, adj)
        x = self.tcn(x) + res
        if self.out_act:
            x = self.act(x)

        return x, adj
