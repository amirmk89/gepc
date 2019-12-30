import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph.graph import Graph
from models.graph.st_graph_conv_block import ConvBlock


class GCAE(nn.Module):
    """
        Graph Conv AutoEncoder
    """
    def __init__(self, in_channels, h_dim=8, graph_args=None, split_seqs=True, eiw=True,
                 dropout=0.0, input_frames=12, conv_oper=None, act=None, headless=False, **kwargs):
        super().__init__()
        # load graph
        if graph_args is None:
            graph_args = {'strategy': 'spatial', 'layout': 'openpose', 'headless': headless}
        self.graph = Graph(**graph_args)
        dec_1st_residual = kwargs.get('dec_1st_residual', None)

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.conv_oper = 'sagc' if conv_oper is None else conv_oper
        self.headless = headless

        # build networks
        num_node = self.graph.num_node
        self.fig_per_seq = 2
        if split_seqs:
            self.fig_per_seq = 1
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.kernel_size = kernel_size
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.dropout = dropout
        self.act = get_act(act)  # Activation function for conv block. Defaults to ReLU

        self.in_channels = in_channels
        self.h_dim = h_dim

        arch_dict = {'enc_ch_fac': [4, 4, 4, 6, 6, 6, 8, 8, 4],
                     'enc_stride': [1, 1, 2, 1, 1, 3, 1, 1, 1],
                     'dec_ch_fac': [4, 8, 8, 6, 6, 6],
                     'dec_stride': [1, 3, 1, 1, 2, 1]}

        self.enc_ch_fac = arch_dict['enc_ch_fac']
        self.enc_stride = arch_dict['enc_stride']
        self.dec_ch_fac = arch_dict['dec_ch_fac']
        self.dec_stride = arch_dict['dec_stride']

        self.out_bn  = kwargs.get('out_bn', False)
        self.out_act = kwargs.get('out_act', False)
        self.out_res = kwargs.get('out_res', False)
        self.gen_ae(self.enc_ch_fac,
                    self.enc_stride,
                    self.dec_ch_fac,
                    self.dec_stride,
                    dec_1st_residual=dec_1st_residual)

        downsample_factor = np.multiply.reduce(np.array(self.enc_stride))
        self.hidden_dim = (input_frames / downsample_factor) * num_node * h_dim * self.fig_per_seq
        self.hidden_dim *= self.enc_ch_fac[-1]

        # Edge weighting
        if eiw and (not conv_oper.startswith('sagc')):
            self.ei_enc = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_enc])
            self.ei_dec = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_dec])
        else:
            self.ei_enc = [1] * len(self.st_gcn_enc)
            self.ei_dec = [1] * len(self.st_gcn_dec)

    def forward(self, x, ret_z=False):
        z, x_size, x_ref = self.encode(x)
        x_reco = self.decode(z, x_size, x_ref)
        if ret_z:
            return x_reco, z
        else:
            return x_reco

    def encode(self, x):
        if self.fig_per_seq == 1:
            if len(x.size()) == 4:
                x = x.unsqueeze(4)
        # Return to (N*M, c, t, v) structure
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_enc, self.ei_enc):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        x = x.contiguous()
        x = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)
        x_ref = x
        x_size = x.size()
        x = x.contiguous()
        x = x.view(N, -1)
        return x, x_size, x_ref

    def decode(self, z, x_size, x_ref=None):
        # Decoding layers
        x = z.view(x_size)
        N, C, T, V, M = x_size
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(N * M, C, T, V)
        for ind, (layer_, importance) in enumerate(zip(self.st_gcn_dec, self.ei_dec)):
            if type(layer_) == ConvBlock:
                x, _ = layer_(x, self.A * importance)  # A graph convolution
            else:
                x = layer_(x)  # An upsampling layer

        x, _ = self.dec_final_gcn(x, self.A * self.ei_dec[-1])  # Final layer has no upsampling
        if self.fig_per_seq == 1:
            return x

        NM, c, t, v = x.size()
        x = x.view(N, M, c, t, v)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x

    def gen_ae(self, enc_ch_fac, enc_stride, dec_ch_fac=None, dec_stride=None, symmetric=True, dec_1st_residual=True):
        if dec_ch_fac is not None or dec_stride is not None:
            symmetric = False
        if symmetric:
            dec_ch_fac = enc_ch_fac[::-1]
            dec_stride = enc_stride[::-1]
        self.build_enc(enc_ch_fac, enc_stride)
        self.build_dec(dec_ch_fac, dec_stride, dec_1st_residual=dec_1st_residual)

    def build_enc(self, enc_ch_fac, enc_stride):
        """
        Generate and encoder according to a series of dimension factors and strides
        """
        if len(enc_ch_fac) != len(enc_stride):
            raise Exception("Architecture error")

        enc_kwargs = [{'dropout': self.dropout, 'conv_oper': self.conv_oper, 'act': self.act,
                       'headless': self.headless} for _ in enc_ch_fac]
        enc_kwargs[0] = {'residual': False, **enc_kwargs[0]}
        enc_kwargs[-1] = {'out_act': False, **enc_kwargs[-1]}  # No Relu for final encoder layer
        st_gcn_enc = [ConvBlock(self.in_channels, enc_ch_fac[0] * self.h_dim, self.kernel_size, enc_stride[0],
                                **enc_kwargs[0])]
        for i in range(1, len(enc_ch_fac)):
            st_gcn_enc.append(
                ConvBlock(enc_ch_fac[i - 1] * self.h_dim, enc_ch_fac[i] * self.h_dim, self.kernel_size, enc_stride[i],
                          **enc_kwargs[i]))
        self.st_gcn_enc = nn.ModuleList(st_gcn_enc)

    def build_dec(self, dec_ch_fac, dec_stride, dec_1st_residual=True):
        if len(dec_ch_fac) != len(dec_stride):
            raise Exception("Architecture error")
        dec_kwargs = [{'dropout': self.dropout, 'conv_oper': self.conv_oper,
                       'act': self.act, 'headless': self.headless, } for _ in dec_ch_fac]
        dec_kwargs[1] = {'residual': dec_1st_residual, **dec_kwargs[1]}
        dec_kwargs += [{'residual': self.out_res, 'out_act': self.out_act, 'out_bn': self.out_bn, **dec_kwargs[0]}]
        st_gcn_dec = []
        for i in range(1, len(dec_ch_fac)):
            if dec_stride[i] != 1:
                st_gcn_dec.append(nn.Upsample(scale_factor=(dec_stride[i], 1), mode='bilinear'))

            st_gcn_dec.append(ConvBlock(dec_ch_fac[i - 1] * self.h_dim, dec_ch_fac[i] * self.h_dim, self.kernel_size, 1))

        # Add output layer back to in_channels w/o relu, bn or residuals
        if dec_kwargs[-1]['conv_oper'].startswith(('sagc')):
            dec_kwargs[-1]['conv_oper'] = 'gcn'
        self.dec_final_gcn = ConvBlock(dec_ch_fac[i] * self.h_dim, self.in_channels, self.kernel_size, 1,
                                       **(dec_kwargs[-1]))
        self.st_gcn_dec = nn.ModuleList(st_gcn_dec)


class Encoder(nn.Module):
    def __init__(self, model):
        super(Encoder, self).__init__()
        self.model = model

    def forward(self, x):
        x, _, _, _ = self.model.encode(x)
        return x


def get_act(act_type):
    if act_type is None:
        return nn.ReLU(inplace=True)
    if act_type.lower() == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type.lower() == 'mish':
        return Mish()


class Mish(nn.Module):
    """
    Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    https://arxiv.org/abs/1908.08681v1
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))
