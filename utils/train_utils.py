import os
import torch
import numpy as np

from utils.clustering import compute_features, Kmeans


def init_clusters(dataset, dcec_args, encoder, num_reevals=0):
    downsample_factor = vars(dcec_args).get('k_init_downsample', 1)
    downsample_data = downsample_factor > 1
    initial_clusters, clustering_loss = calc_initial_clusters(dcec_args, encoder, dataset['train'],
                                                              downsample_data=downsample_data)
    if dcec_args.epochs > 0:  # To avoid unneeded calculation when clustering not used
        for i in range(num_reevals):
            curr_clusters, curr_clustering_loss = calc_initial_clusters(dcec_args, encoder, dataset['train'],
                                                                        downsample_data=downsample_data)
            if curr_clustering_loss < clustering_loss:
                clustering_loss = curr_clustering_loss
                initial_clusters = curr_clusters
    hidden_dim = initial_clusters.shape[1]
    return hidden_dim, initial_clusters


def calc_initial_clusters(args, encoder, dataset, concat_vid=False,
                          shuffle_loader=True, downsample_data=False):
    """
    When no initial clusters supplied, uses deep cluster to calculate initial K-Means
    :returns a (k, num_features) centroid matrix
    """
    if downsample_data:
        indices = np.random.permutation(len(dataset))[::args.k_init_downsample]
        kmeans_dataset = torch.utils.data.Subset(dataset, indices)
        kmeans_dataset.dataset.return_indices = False
    else:
        kmeans_dataset = dataset
        kmeans_dataset.return_indices = False
    loader_args = {'batch_size': args.k_init_batch, 'num_workers': args.num_workers}
    kmeans_loader = torch.utils.data.DataLoader(kmeans_dataset, shuffle=shuffle_loader, **loader_args)
    features = compute_features(kmeans_loader, encoder, args=args, concat_vid=concat_vid)
    deepcluster = Kmeans(args.n_clusters, knn=1)
    clustering_loss, _ = deepcluster.cluster(features, verbose=True)
    return deepcluster.centroids, clustering_loss


def calc_reg_loss(model, reg_type='l2', avg=True):
    reg_loss = None
    parameters = list(param for name, param in model.named_parameters() if 'bias' not in name)
    num_params = len(parameters)
    if reg_type.lower() == 'l2':
        for param in parameters:
            if reg_loss is None:
                reg_loss = 0.5 * torch.sum(param ** 2)
            else:
                reg_loss = reg_loss + 0.5 * param.norm(2) ** 2

        if avg:
            reg_loss /= num_params
        return reg_loss
    else:
        return torch.tensor(0.0, device=model.device)


def get_fn_suffix(args):
    patch_str = '_patch' if args.patch_features else ''
    fn_suffix = 'stc_' + args.conv_oper + patch_str
    return fn_suffix


def csv_log_dump(args, log_dict):
    """
    Create CSV log line, with the following format:
    Date, Time, Seed, conv_oper, n_transform, norm_scale, prop_norm_scale, seg_stride, seg_len, patch_features,
    patch_size, optimizer, dropout, ae_batch_size, ae_epochs, ae_lr, ae_lr_decay, ae_lr_decay, ae_wd,
    F ae loss, K (=num_clusters), dcec_batch_size, dcec_epochs, dcec_lr_decay, dcec_lr, dcec_lr_decay,
    alpha (=L2 reg coef), gamma, update_interval, F Delta Labels, F dcec loss
    :return:
    """
    try:
        date_time = args.ckpt_dir.split('/')[-3]  # 'Aug20_0839'
        date_str, time_str = date_time.split('_')[:2]
    except:
        date_time = 'parse_fail'
        date_str, time_str = '??', '??'
    param_arr = [date_str, time_str, args.seed, args.conv_oper, args.num_transform, args.norm_scale,
                 args.prop_norm_scale, args.seg_stride, args.seg_len, args.patch_features, args.patch_size,
                 args.ae_optimizer, args.ae_sched, args.dropout, args.ae_batch_size,
                 args.ae_epochs, args.ae_lr, args.ae_lr_decay, args.ae_weight_decay, log_dict['F_ae_loss'],
                 args.n_clusters, args.dcec_batch_size, args.dcec_epochs, args.dcec_optimizer, args.dcec_sched,
                 args.dcec_lr, args.dcec_lr_decay, args.alpha, args.gamma, args.update_interval,
                 log_dict['F_delta_labels'], log_dict['F_dcec_loss'], log_dict['dp_auc'], args.headless]

    res_str = '_{}'.format(int(10 * log_dict['dp_auc']))
    log_template = len(param_arr) * '{}, ' + '\n'
    log_str = log_template.format(*param_arr)
    debug_str = '_debug' if args.debug else ''
    csv_path = os.path.join(args.ckpt_dir, '{}{}{}_log_dump.csv'.format(date_time, debug_str, res_str))
    with open(csv_path, 'w') as csv_file:
        csv_file.write(log_str)


