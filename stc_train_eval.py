import os
import random
import collections
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.gcae.gcae import Encoder
from models.fe.fe_model import init_fenet

from models.dc_gcae.dc_gcae import DC_GCAE, load_ae_dcec
from models.dc_gcae.dc_gcae_training import dc_gcae_train
from models.gcae.gcae_training import Trainer

from utils.data_utils import ae_trans_list
from utils.train_utils import get_fn_suffix, init_clusters
from utils.train_utils import csv_log_dump
from utils.scoring_utils import dpmm_calc_scores, score_dataset, avg_scores_by_trans
from utils.pose_seg_dataset import PoseSegDataset
from utils.pose_ad_argparse import init_stc_parser, init_stc_sub_args
from utils.optim_utils.optim_init import init_optimizer, init_scheduler


def main():
    parser = init_stc_parser()
    args = parser.parse_args()
    log_dict = collections.defaultdict(int)

    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    args, ae_args, dcec_args, res_args = init_stc_sub_args(args)
    print(args)

    dataset, loader = get_dataset_and_loader(ae_args)

    ae_fn = vars(args).get('ae_fn', None)
    dcec_fn = vars(args).get('dcec_fn', None)

    if dcec_fn:  # Load pretrained models
        pretrained = True
        dc_gcae = load_ae_dcec(dcec_fn)
        args.ae_fn = dcec_fn.split('/')[-1]
        res_args.ae_fn = dcec_fn.split('/')[-1]
    else:
        pretrained = False
        if ae_fn:  # Load pretrained AE and train DCEC
            fe_model = init_fenet(args)
            fe_model.load_checkpoint(ae_fn)
        else:  # Train an AE
            backbone = 'resnet' if args.patch_features else None
            model = init_fenet(args, backbone)

            loss = nn.MSELoss()
            ae_optimizer_f = init_optimizer(args.ae_optimizer, lr=args.ae_lr)
            ae_scheduler_f = init_scheduler(args.ae_sched, lr=args.ae_lr, epochs=args.ae_epochs)
            trainer = Trainer(ae_args, model, loss, loader['train'], loader['test'], optimizer_f=ae_optimizer_f,
                              scheduler_f=ae_scheduler_f, fn_suffix=get_fn_suffix(args))
            ae_fn, log_dict['F_ae_loss'] = trainer.train(checkpoint_filename=ae_fn, args=ae_args)
            args.ae_fn = dcec_args.ae_fn = res_args.ae_fn = ae_fn
            fe_model = trainer.model

        # Train DCEC models
        encoder = Encoder(model=fe_model).to(args.device)
        hidden_dim, initial_clusters = init_clusters(dataset, dcec_args, encoder)
        dc_gcae = DC_GCAE(fe_model, hidden_dim, n_clusters=args.n_clusters, initial_clusters=initial_clusters)
        _, log_dict['F_delta_labels'], log_dict['F_dcec_loss'] = dc_gcae_train(dc_gcae, dataset['train'], dcec_args)

    # Normality scoring phase
    dc_gcae.eval()
    if pretrained and getattr(args, 'dpmm_fn', False):
        pt_dpmm = args.dpmm_fn
    else:
        pt_dpmm = None

    dp_scores, gt, metadata = dpmm_calc_scores(dc_gcae, dataset['train'], dataset['test'],
                                               args=res_args, ret_metadata=True, pt_dpmm_path=pt_dpmm)

    dp_scores_tavg, _ = avg_scores_by_trans(dp_scores, gt, args.num_transform)
    max_clip = 5 if args.debug else None
    dp_auc, dp_shift, dp_sigma = score_dataset(dp_scores_tavg, metadata, max_clip=max_clip)

    # Logging and recording results
    print("Done with {} AuC for {} samples and {} trans".format(dp_auc, dp_scores_tavg.shape[0], args.num_transform));
    log_dict['dp_auc'] = 100 * dp_auc
    csv_log_dump(args, log_dict)


def get_dataset_and_loader(args):
    patch_size = int(args.patch_size)
    if args.patch_db:
        patch_suffix_str = 'ing{}x{}.lmdb'.format(patch_size, patch_size)
        patch_size = (patch_size, patch_size)
        patch_db_path = {k: os.path.join(v, k+patch_suffix_str) for k, v in args.vid_path.items()}
    else:
        patch_db_path = {k: None for k, v in args.vid_path.items()}

    trans_list = ae_trans_list[:args.num_transform]

    dataset_args = {'transform_list': trans_list, 'debug': args.debug, 'headless': args.headless,
                    'scale': args.norm_scale, 'scale_proportional': args.prop_norm_scale, 'seg_len': args.seg_len,
                    'patch_size': patch_size, 'return_indices': True, 'return_metadata': True}

    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    dataset, loader = dict(), dict()
    for split in ['train', 'test']:
        dataset_args['seg_stride'] = args.seg_stride if split is 'train' else 1  # No strides for test set
        dataset_args['train_seg_conf_th'] = args.train_seg_conf_th if split is 'train' else 0.0
        if args.patch_features:
            dataset[split] = PoseSegDataset(args.pose_path[split], args.vid_path[split], patch_db_path[split],
                                            **dataset_args)
        else:
            dataset[split] = PoseSegDataset(args.pose_path[split], **dataset_args)
        loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))
    return dataset, loader


def save_result_npz(args, scores, scores_tavg, metadata, sfmax_maxval, auc, dp_auc=None):
    debug_str = '_debug' if args.debug else ''
    auc_int = int(1000 * auc)
    dp_auc_str = ''
    if dp_auc is not None:
        dp_auc_int = int(1000 * dp_auc)
        dp_auc_str = '_dp{}'.format(dp_auc_int)
    auc_str = '_{}'.format(auc_int)
    res_fn = args.ae_fn.split('.')[0] + '_res{}{}{}.npz'.format(dp_auc_str, auc_str, debug_str)
    res_path = os.path.join(args.ckpt_dir, res_fn)
    np.savez(res_path, scores=scores, sfmax_maxval=sfmax_maxval, args=args, metadata=metadata,
             scores_tavg=scores_tavg, dp_best=dp_auc)


if __name__ == '__main__':
    main()

