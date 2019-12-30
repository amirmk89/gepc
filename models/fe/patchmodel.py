import os
import torch
import torch.nn as nn

from models.gcae.gcae import GCAE
from models.fe.patch_resnet import pt_resnet


class PatchModel(nn.Module):
    """
    A Wrapper class for hadling per-patch feature extraction
    """
    def __init__(self, patch_fe, gcae, backbone='resnet'):
        super().__init__()
        self.backbone = backbone
        self.patch_fe = patch_fe
        self.outdim = getattr(self.patch_fe, 'outdim', 3)
        self.gcae = gcae

    def forward(self, x):
        input_feature_graph = self.extract_patch_features(x)
        z, x_size = self.graph_encode(input_feature_graph)
        reco_graph = self.decode(z, x_size)
        return reco_graph, input_feature_graph

    def encode(self, x):
        feature_graph = self.extract_patch_features(x)
        z, x_size, _ = self.gcae.encode(feature_graph)
        return z, x_size, x, feature_graph

    def graph_encode(self, input_feature_graph):
        z, x_size, _ = self.gcae.encode(input_feature_graph)
        return z, x_size

    def decode(self, z, x_size):
        x = self.gcae.decode(z, x_size)
        return x

    def extract_patch_features(self, x):
        # Take a [N, C, T, V, W, H] tensor,
        # permute and view as [N*T*V, C_in, W, H]
        # Apply models
        # Return feature shape of [N, C_new, T, V]
        if self.backbone is None:  # Using keypoints only, w/o patches
            return x
        else:
            n, c, t, v, w, h = x.size()
            x_perm = x.permute(0, 2, 3, 1, 4, 5).contiguous()
            x_perm = x_perm.view(n * t * v, c, w, h)
            f_perm = self.patch_fe(x_perm)  # y is [N*T*V, C_new]
            f_perm = f_perm.view(n, t, v, -1)
            feature_graph = f_perm.permute(0, 3, 1, 2).contiguous()
            return feature_graph

    def get_patchmodel_dict(self, epoch, args=None, optimizer=None):
        state = {
            'epoch': epoch + 1,
            'outdim': self.outdim,
            'backbone': self.backbone,
            'patch_model': self.patch_fe.state_dict(),
            'gcae': self.gcae.state_dict(),
            'args': args,
        }
        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
        if hasattr(self.gcae, 'num_class'):
            state['n_classes'] = self.gcae.num_class
        if hasattr(self.gcae, 'h_dim'):
            state['h_dim'] = self.gcae.h_dim
        return state

    def save_checkpoint(self, epoch, args=None, optimizer=None, filename=None):
        state = self.get_patchmodel_dict(epoch, args=args, optimizer=optimizer)
        path_join = os.path.join(args.ckpt_dir, filename)
        torch.save(state, path_join)

    def load_checkpoint(self, path):
        try:
            patchmodel_dict = torch.load(path)
            self.load_patchmodel_dict(patchmodel_dict)
            print("Checkpoint loaded successfully from '{}')\n" .format(path))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'.".format(path))

    def load_patchmodel_dict(self, patchmodel_dict, backbone=None, args=None):
        if args is None:
            args = patchmodel_dict['args']
        self.backbone = patchmodel_dict.get('backbone', backbone)
        self.patch_fe = pt_resnet(backbone=self.backbone)

        fe_state_dict = patchmodel_dict.get('patch_model', patchmodel_dict)
        gcae_state_dict = patchmodel_dict.get('gcae', patchmodel_dict)

        if backbone is not None:
            self.patch_fe.load_state_dict(fe_state_dict)

        in_channels = getattr(self.patch_fe, 'outdim', 3)
        headless = args.headless
        self.gcae = GCAE(in_channels,
                         dropout=args.dropout,
                         conv_oper=args.conv_oper,
                         act=args.act,
                         headless=headless)
        self.gcae.load_state_dict(gcae_state_dict)

