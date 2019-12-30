import os
import time
import shutil
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.train_utils import calc_reg_loss
from models.dc_gcae.dc_gcae_training import adjust_lr


class Trainer:
    def __init__(self, args, model, loss, train_loader, test_loader,
                 optimizer_f=None, scheduler_f=None, fn_suffix=''):
        self.model = model
        self.args = args
        self.args.start_epoch = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.fn_suffix = fn_suffix  # For checkpoint filename
        # Loss, Optimizer and Scheduler
        self.loss = loss
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        else:
            return optim.SGD(self.model.parameters(), lr=self.args.lr,)

    def adjust_lr(self, epoch):
        return adjust_lr(self.optimizer, epoch, self.args.lr, self.args.lr_decay, self.scheduler)

    def save_checkpoint(self, epoch, args, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = args

        path_join = os.path.join(self.args.ckpt_dir, filename)
        torch.save(state, path_join)
        if is_best:
            shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))

    def load_checkpoint(self, filename):
        filename = self.args.ckpt_dir + filename
        try:
            checkpoint = torch.load(filename)
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.args.ckpt_dir, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))

    def train(self, num_epochs=None, log=True, checkpoint_filename=None, args=None):
        time_str = time.strftime("%b%d_%H%M_")
        if checkpoint_filename is None:
            checkpoint_filename = time_str + self.fn_suffix + '_checkpoint.pth.tar'
        if num_epochs is None:  # For manually setting number of epochs, i.e. for fine tuning
            start_epoch = self.args.start_epoch
            num_epochs = self.args.epochs
        else:
            start_epoch = 0

        self.model.train()
        self.model = self.model.to(args.device)
        for epoch in range(start_epoch, num_epochs):
            loss_list = []
            ep_start_time = time.time()
            print("Started epoch {}".format(epoch))
            for itern, data_arr in enumerate(tqdm(self.train_loader)):
                data = data_arr[0].to(args.device, non_blocking=True)
                output, reco_data = self.model(data)

                reco_loss = self.loss(output, reco_data)

                reg_loss = calc_reg_loss(self.model)
                loss = reco_loss + 1e-3 * args.alpha * reg_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_list.append(loss.item())

            print("Epoch {0} done, loss: {1:.7f}, took: {2:.3f}sec".format(epoch, np.mean(loss_list),
                                                                           time.time()-ep_start_time))
            new_lr = self.adjust_lr(epoch)
            print('lr: {0:.3e}'.format(new_lr))

            self.model.save_checkpoint(epoch, args=args, filename=checkpoint_filename)

        return checkpoint_filename, np.mean(loss_list)

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        if hasattr(self.model, 'num_class'):
            checkpoint_state['n_classes'] = self.model.num_class
        if hasattr(self.model, 'h_dim'):
            checkpoint_state['h_dim'] = self.model.h_dim
        return checkpoint_state

    def test(self, cur_epoch, ret_sfmax=False, log=True, args=None):
        self._test(cur_epoch, self.test_loader, ret_sfmax=ret_sfmax, log=log, args=args)

    def _test(self, cur_epoch, test_loader, ret_sfmax=True, log=True, args=None):
        print("Testing")
        self.model.eval()
        test_loss = 0
        output_arr = []
        for itern, data_arr in enumerate(test_loader):
            # Get Data
            with torch.no_grad():
                data = data_arr[0].to(args.device)
                output = self.model(data)

            if ret_sfmax:
                output_sfmax = output
                output_arr.append(output_sfmax.detach().cpu().numpy())
                del output_sfmax

            loss = self.loss(output, data)
            test_loss += loss.item()

        test_loss /= len(test_loader.dataset)
        print("--> Test set loss {:.7f}".format(test_loss))
        self.model.train()
        if ret_sfmax:
            return output_arr

