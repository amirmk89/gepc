"""
A Deep clustering models using a Graph-Convolutional Auto Encoder
Takes a trained GCAE, adds a classification layer and loss and optimizes for clustering performance
while fine-tuning the Autoencoder.

"""
import os
import torch
import torch.nn as nn

from models.fe.fe_model import init_fenet
from models.fe.patchmodel import PatchModel
from models.dc_gcae.clustering_layer import ClusteringLayer


class DC_GCAE(nn.Module):
    def __init__(self, gcae, input_shape,
                 n_clusters=10, alpha=1.0,
                 initial_clusters=None, device='cuda:0'):
        super(DC_GCAE, self).__init__()
        self.input_shape = input_shape
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.y_pred = []

        # Define GCAE, later to be loaded with pretrained weights
        self.gcae = gcae.to(device)
        self.encoder = self.gcae.encode
        self.decoder = self.gcae.decode

        # Define Deep Clustering models
        self.clustering_layer = ClusteringLayer(self.n_clusters, input_shape, weights=initial_clusters)

    def forward(self, x_in, ret_z=False):

        x = x_in
        enc_ret = self.encoder(x)
        z, x_size, feature_graph = enc_ret[0], enc_ret[1], enc_ret[-1]
        x_reco = self.decoder(z, x_size)
        cls_sfmax = self.clustering_layer(z)

        if ret_z:
            return cls_sfmax, x_reco, feature_graph, z
        else:
            return cls_sfmax, x_reco, feature_graph

    def predict(self, x):
        z, x_size, _ = self.encoder(x)
        cls_sfmax = self.clustering_layer(z)
        cls = torch.argmax(cls_sfmax, 1)
        return cls

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        w = weight.t() / weight.sum(1)
        w = w.t()
        return w


def save_checkpoint(model, path, args, filename=None):
    if filename is None:
        filename = 'dcec_checkpoint.pth.tar'
    path_join = os.path.join(path, filename)

    outdim = getattr(model.gcae, 'outdim', 3)
    if isinstance(model.gcae, PatchModel):
        pm_state = model.gcae.get_patchmodel_dict(0, args=args)
    else:
        pm_state = {'state_dict': model.gcae.state_dict()}
        if hasattr(model.gcae, 'h_dim'):
            pm_state['h_dim'] = model.gcae.h_dim

    state = {'args': args,
             'outdim': outdim,
             'state_dict': pm_state,
             'n_clusters': model.clustering_layer.n_clusters,
             'clustering_layer': model.clustering_layer.state_dict(), }
    torch.save(state, path_join)


def load_ae_dcec(model_path):
    """
    Loading function for models including the PatchModel and clustering layer
    :param model_path: Path to models file containing the dictionaries
    :return:
    """
    model_dict = torch.load(model_path)
    args = model_dict['args']
    n_clusters = model_dict.get('n_clusters', 10)
    backbone = 'resnet' if args.patch_features else None

    kwargs = {}  # Add final layer residual and batchnorm if in weight dictionary

    gcae = init_fenet(args, backbone, **kwargs)
    gcae.load_patchmodel_dict(model_dict['state_dict'], backbone, args=args)

    clustering_dim = model_dict['clustering_layer']['clusters'].size(1)
    model = DC_GCAE(gcae, clustering_dim, n_clusters)
    model.gcae.load_state_dict(model_dict['state_dict'], strict=False)
    model.clustering_layer.load_state_dict(model_dict['clustering_layer'])
    return model


