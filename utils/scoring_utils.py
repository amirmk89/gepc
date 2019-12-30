import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from sklearn import mixture
from joblib import dump, load


def dpmm_calc_scores(model, train_dataset, eval_normal_dataset, eval_abn_dataset=None, args=None,
                     ret_metadata=False, dpmm_components=10, dpmm_downsample_fac=10, pt_dpmm_path=None):
    """
    Wrapper for extracting features for DNS experiment, given a trained DCEC models, a normal training dataset and two
    datasets for evaluation, a "normal" one and an "abnormal" one
    :param model: A trained model
    :param train_dataset: "normal" training dataset, for alpha calculation
    :param eval_normal_dataset: "normal" or "mixed" evaluation dataset
    :param eval_abn_dataset: "abnormal" evaluation dataset (optional)
    :param args - command line arguments
    :param ret_metadata:
    :param dpmm_components:  Truncation parameter for DPMM
    :param dpmm_downsample_fac: Downsampling factor for DPMM fitting
    :param pt_dpmm_path: Path to a pretrained DPMM model
    :return actual experiment done after feature extraction (calc_p)
    """
    # Alpha calculation and fitting
    train_p = calc_p(model, train_dataset, args, ret_metadata=False)
    eval_p_ret = calc_p(model, eval_normal_dataset, args, ret_metadata=ret_metadata)
    if ret_metadata:
        eval_p_normal, metadata = eval_p_ret
    else:
        eval_p_normal = eval_p_ret

    p_vec = eval_p_normal
    eval_p_abn = None
    if eval_abn_dataset:
        eval_p_abn = calc_p(model, eval_abn_dataset, args, ret_metadata=ret_metadata)
        p_vec = np.concatenate([eval_p_normal, eval_p_abn])

    print("Started fitting DPMM")
    if pt_dpmm_path is None:
        dpmm_mix = mixture.BayesianGaussianMixture(n_components=dpmm_components,
                                                   max_iter=500, verbose=1, n_init=1)
        dpmm_mix.fit(train_p[::dpmm_downsample_fac])
    else:
        dpmm_mix = load(pt_dpmm_path)

    dpmm_scores = dpmm_mix.score_samples(p_vec)

    if eval_p_abn is not None:
        gt = np.concatenate([np.ones(eval_p_normal.shape[0], dtype=np.int),
                             np.zeros(eval_p_abn.shape[0], dtype=np.int)])
    else:
        gt = np.ones_like(dpmm_scores, dtype=np.int)

    try:  # Model persistence
        dpmm_fn = args.ae_fn.split('.')[0] + '_dpgmm.pkl'
        dpmm_path = os.path.join(args.ckpt_dir, dpmm_fn)
        dump(dpmm_mix, dpmm_path)
    except ModuleNotFoundError:
        print("Joblib missing, DPMM not saved")

    if ret_metadata:
        return dpmm_scores, gt, metadata
    else:
        return dpmm_scores, gt


def calc_p(model, dataset, args, ret_metadata=True, ret_z=False):
    """ Evalutates the models output over the data in the dataset. """
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, drop_last=False, pin_memory=True)
    model = model.to(args.device)
    model.eval()
    p = p_compute_features(loader, model, device=args.device, ret_z=ret_z)

    if ret_z:
        p, z = p
        if ret_metadata:
            return p, z, dataset.metadata
        else:
            return p, z
    else:
        if ret_metadata:
            return p, dataset.metadata
        else:
            return p


def p_compute_features(loader, model, device='cuda:0', ret_z=False):
    sfmax = []
    z_arr = []
    for itern, data_arr in enumerate(loader):
        data = data_arr[0]
        if itern % 100 == 0:
            print("Compute Features Iter {}".format(itern))
        with torch.no_grad():
            data = data.to(device)
            model_ret = model(data, ret_z=ret_z)
            cls_sfmax = model_ret[0]
            if ret_z:
                z = model_ret[-1]
                z_arr.append(z.to('cpu', non_blocking=True).numpy().astype('float32'))

            cls_sfmax = torch.reshape(cls_sfmax, (cls_sfmax.size(0), -1))
            sfmax.append(cls_sfmax.to('cpu', non_blocking=True).numpy().astype('float32'))

    sfmax = np.concatenate(sfmax)
    if ret_z:
        z_arr = np.concatenate(z_arr)
        return sfmax, z_arr
    else:
        return sfmax


def score_dataset(score_vals, metadata, max_clip=None, scene_id=None):
    gt_arr, scores_arr, score_ids_arr, metadata_arr = get_dataset_scores(score_vals, metadata, max_clip, scene_id)
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    auc, shift, sigma = score_align(scores_np, gt_np)
    return auc, shift, sigma


def get_dataset_scores(scores, metadata, max_clip=None, scene_id=None):
    dataset_gt_arr = []
    dataset_scores_arr = []
    dataset_metadata_arr = []
    dataset_score_ids_arr = []
    metadata_np = np.array(metadata)
    per_frame_scores_root = 'data/testing/test_frame_mask/'
    clip_list = os.listdir(per_frame_scores_root)
    clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    if scene_id is not None:
        clip_list = [cl for cl in clip_list if int(cl[:2]) == scene_id]
    if max_clip:
        max_clip = min(max_clip, len(clip_list))
        clip_list = clip_list[:max_clip]
    print("Scoring {} clips".format(len(clip_list)))
    for clip in clip_list:
        clip_res_fn = os.path.join(per_frame_scores_root, clip)
        clip_gt = np.load(clip_res_fn)
        scene_id, clip_id = [int(i) for i in clip.split('.')[0].split('_')]
        clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                      (metadata_np[:, 0] == scene_id))[0]
        clip_metadata = metadata[clip_metadata_inds]
        clip_fig_idxs = set([arr[2] for arr in clip_metadata])
        scores_zeros = np.zeros(clip_gt.shape[0])
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}
        for person_id in clip_fig_idxs:
            person_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                            (metadata_np[:, 0] == scene_id) &
                                            (metadata_np[:, 2] == person_id))[0]
            pid_scores = scores[person_metadata_inds]
            pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds])
            clip_person_scores_dict[person_id][pid_frame_inds] = pid_scores

        clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
        clip_score = np.amax(clip_ppl_score_arr, axis=0)
        fig_score_id = [list(clip_fig_idxs)[i] for i in np.argmax(clip_ppl_score_arr, axis=0)]
        dataset_gt_arr.append(clip_gt)
        dataset_scores_arr.append(clip_score)
        dataset_score_ids_arr.append(fig_score_id)
        dataset_metadata_arr.append([scene_id, clip_id])
    return dataset_gt_arr, dataset_scores_arr, dataset_score_ids_arr, dataset_metadata_arr


def score_align(scores_np, gt, seg_len=12, sigma=40):
    scores_shifted = np.zeros_like(scores_np)
    shift = seg_len + (seg_len // 2) - 1
    scores_shifted[shift:] = scores_np[:-shift]
    scores_smoothed = gaussian_filter1d(scores_shifted, sigma)
    auc = roc_auc_score(gt, scores_smoothed)
    return auc, shift, sigma


def avg_scores_by_trans(scores, gt, num_transform=5, ret_first=False):
    score_mask, scores_by_trans, scores_tavg = dict(), dict(), dict()
    gti = {'normal': 1, 'abnormal': 0}
    for k, gt_val in gti.items():
        score_mask[k] = scores[gt == gt_val]
        scores_by_trans[k] = score_mask[k].reshape(-1, num_transform)
        scores_tavg[k] = scores_by_trans[k].mean(axis=1)

    gt_trans_avg = np.concatenate([np.ones_like(scores_tavg['normal'], dtype=np.int),
                                   np.zeros_like(scores_tavg['abnormal'], dtype=np.int)])
    scores_trans_avg = np.concatenate([scores_tavg['normal'], scores_tavg['abnormal']])
    if ret_first:
        scores_first_trans = dict()
        for k, v in scores_by_trans.items():
            scores_first_trans[k] = v[:, 0]
        scores_first_trans = np.concatenate([scores_first_trans['normal'], scores_first_trans['abnormal']])
        return scores_trans_avg, gt_trans_avg, scores_first_trans
    else:
        return scores_trans_avg, gt_trans_avg
