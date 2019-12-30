import os
import math
import time
import numpy as np
import torch
from torch import nn as nn
from tqdm import tqdm

from models.dc_gcae.dc_gcae import save_checkpoint
from utils.train_utils import calc_reg_loss


def dc_gcae_train(dc_gcae, dataset, args, optimizer=None, scheduler=None, stop_cret=1e-3):
    """
    By now, the following should have been completed:
    Step 1: Pretraining the AE
    Step 2: Initializing Clusters with K-Means
    Now:
    Step 3: Deep Clustering
    """
    params = list(dc_gcae.parameters()) + list(dc_gcae.clustering_layer.parameters())
    if optimizer is None:
        new_lr, optimizer = get_optimizer(args, optimizer, params)
    else:
        new_lr, optimizer = args.lr, optimizer(params)

    ckpt_dir, ckpt_filename = get_ckpt_path(args)
    args.ckpt_filename = ckpt_filename

    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True, }
    dataset.return_indices = True
    e_loader = torch.utils.data.DataLoader(dataset, shuffle=False, **loader_args)
    m_loader = torch.utils.data.DataLoader(dataset, shuffle=True, drop_last=True, **loader_args)

    reco_loss_fn = nn.MSELoss(size_average=True).to(args.device)
    cls_loss_fn = nn.KLDivLoss(size_average=False).to(args.device)

    epoch = 0
    y_pred = []
    y_pred_prev = np.copy(y_pred)
    loss = [0, 0, 0]
    p = None
    convergence_str = ''

    dc_gcae = dc_gcae.to(args.device)

    optimizer.zero_grad()
    gamma = 0.0
    delta_label = 0.0
    loss_list = []
    stop_flag = False

    eval_epochs, eval_iters = get_eval_epochs_iters(args, m_loader)

    for epoch in range(args.epochs):
        loss_list = []
        if epoch == args.pretrain_epochs:
            print("Pretraining done at epoch {}, applying gamma {}".format(epoch, gamma))
        start_time = time.time()
        for itern, data_arr in enumerate(tqdm(m_loader, desc="Train Epoch", leave=True)):
            if (epoch % eval_epochs == 0 and epoch > args.pretrain_epochs) or (epoch == 0):
                if itern % eval_iters == 0:
                    # E step - Eval distribution and clustering
                    print("Target distribution update at epoch {} iteration {}".format(epoch, itern))
                    p, y_pred = calc_curr_p(dc_gcae, e_loader)

                    # Eval Clustering Performance and check stopping critetion
                    if epoch >= 1:
                        stop_flag, y_pred_prev, delta_label = eval_clustering_stop_cret(y_pred, y_pred_prev,
                                                                                        stop_cret=stop_cret)
                        if epoch >= (args.pretrain_epochs + 3) and stop_flag:
                            print("Stop flag in epoch {}".format(epoch))
                            convergence_str = 'c'
                            break  # Training is Done
                        else:
                            stop_flag = False  # Remove to allow further iterations

        # Train for one epoch: M step
            if not stop_flag:
                indices = data_arr[-1]
                p_iter = torch.from_numpy(p[indices]).to(args.device)
                data = data_arr[0].to(args.device)
                cls_sfmax, x_reco, feature_graph = dc_gcae(data)
                reco_loss = reco_loss_fn(x_reco, feature_graph)
                clustering_loss = cls_loss_fn(torch.log(p_iter), cls_sfmax)

                reg_loss = calc_reg_loss(dc_gcae)
                loss = reco_loss + args.gamma * clustering_loss + args.alpha * reg_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_list.append(loss.item())

        new_lr = adjust_lr(optimizer, epoch, new_lr, args.lr_decay, scheduler=scheduler)
        save_checkpoint(dc_gcae, ckpt_dir, args=args, filename=ckpt_filename)
        print("Epoch {} Done in {}s, loss is {}\n".format(epoch, time.time()-start_time, loss))
        if stop_flag:
            break

    done_str = 'done{}{}_'.format(epoch+1, convergence_str)
    save_checkpoint(dc_gcae, ckpt_dir, args=args, filename=done_str + ckpt_filename)
    return dc_gcae, delta_label, np.mean(loss_list)


def get_optimizer(args, opt, params):
    if opt is None:
        opt = torch.optim.Adam

    optimizer = opt(params, lr=args.lr, weight_decay=args.weight_decay)
    new_lr = args.lr  # Adam Default
    return new_lr, optimizer


def get_ckpt_path(args):
    ckpt_filename = get_ckpt_fn(args)
    ckpt_dir = get_ckpt_dir(args)
    return ckpt_dir, ckpt_filename


def get_ckpt_dir(args):
    ae_model_name = args.ae_fn.split('.')[0]
    ckpt_dir = os.path.join(args.ckpt_dir, ae_model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def get_ckpt_fn(args):
    gamma_str = str(args.gamma).replace('.', '')
    ckpt_filename = time.strftime("%b%d_%H%M") + '_dcec{}_{}_checkpoint.pth.tar'.format(args.n_clusters, gamma_str)
    return ckpt_filename


def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


def get_eval_epochs_iters(args, m_loader):
    epoch_iters = len(m_loader.dataset) // m_loader.batch_size
    eval_frac, eval_intp = math.modf(args.update_interval)
    eval_epochs = int(eval_intp)
    eval_iters = int(eval_frac * epoch_iters) + 1  # Round up to avoid eval at last iter
    # Handling corner cases for mid iteration evaluation:
    if eval_epochs == 0:
        eval_epochs = 1  # Eval every epoch
    if eval_iters == 1:
        eval_iters = epoch_iters + 1  # Once every evaluation epoch
    return eval_epochs, eval_iters


def calc_curr_p(dc_gcae, data_loader, data_ind=0, device='cuda:0'):
    #  Eval new P for the entire dataset
    p = []
    y_pred = []
    for itern, data_arr in enumerate(tqdm(data_loader, desc="P Calculation")):
        with torch.no_grad():
            pose_data = data_arr[data_ind].to(device)
            curr_q, _, _ = dc_gcae(pose_data)
            curr_p = dc_gcae.target_distribution(curr_q)
            y_pred_curr = torch.argmax(curr_q, 1)
            p.append(curr_p.cpu().numpy())
            y_pred.append(y_pred_curr.cpu().numpy())

    p = np.concatenate(p, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return p, y_pred


def eval_clustering_stop_cret(y_pred, y_pred_prev, stop_cret=1e-3):
    # Eval stopping criterion
    stop_flag = False
    delta_label = np.sum(y_pred != y_pred_prev).astype(np.float32) / y_pred.shape[0]
    print('delta_label ', delta_label)
    y_pred_prev = np.copy(y_pred)
    if delta_label < stop_cret:
        print('delta_label ', delta_label, '< tol ', stop_cret)
        print('Reached tolerance threshold. Stopping training if past min epochs.')
        stop_flag = True
    return stop_flag, y_pred_prev, delta_label
