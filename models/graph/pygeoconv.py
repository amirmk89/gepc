import torch
import torch.nn as nn
from models.graph.sagc import SAGC


class PyGeoConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 conv_oper='sagc',
                 headless=False,
                 dropout=0.0,
                 ):
        super().__init__()
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.multi_adj = kernel_size > 1

        if conv_oper == 'sagc':
            self.g_conv = SAGC(in_channels, out_channels, headless=headless)
        elif conv_oper == 'gcn':
            self.g_conv = ConvTemporalGraphical(in_channels, out_channels, kernel_size)
        else:
            raise Exception('No such Conv oper')

    def forward(self, inp, adj):
        return self.g_conv(inp, adj)


class ConvTemporalGraphical(nn.Module):

    """
    The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            stride=(t_stride, 1),
            dilation=(1, 1),
            bias=bias)

    def forward(self, x, adj):
        assert adj.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, adj))

        return x.contiguous(), adj
