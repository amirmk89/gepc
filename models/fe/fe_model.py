from models.gcae.gcae import GCAE
from models.fe.patchmodel import PatchModel
from models.fe.patch_resnet import pt_resnet


def init_fenet(args, backbone='resnet', rm_linear=True, split_seqs=True, **kwargs):
    """
    Initialize the whole feature extraction models, including patch feature extractor (if used) and graph AE
    :param args:
    :param backbone:
    :param rm_linear:
    :param split_seqs:
    :param kwargs:
    :return:
    """
    patch_fe = pt_resnet(backbone=backbone, rm_linear=rm_linear)
    in_channels = getattr(patch_fe, 'outdim', 3)
    headless = getattr(args, 'headless', False)
    graph_args = None
    gcae = GCAE(in_channels,
                graph_args=graph_args,
                dropout=args.dropout,
                conv_oper=args.conv_oper,
                act=args.act,
                headless=headless,
                split_seqs=split_seqs,
                **kwargs)
    return PatchModel(patch_fe, gcae, backbone=backbone)


