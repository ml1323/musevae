import torch.optim as optim
from data.loader import data_loader
from util import *


from unet.unet import Unet
import numpy as np
import torch.nn.functional as F
from data.nuscenes.config import Config
from data.nuscenes_dataloader import data_generator

class Solver(object):

    ####
    def __init__(self, args):

        self.args = args

        self.name = '%s_%s_lr_%s_n_goal_%s_run_%s' % \
                    (args.dataset_name, args.model_name, args.lr, args.num_goal, args.run_id)

        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.device = args.device
        self.dt=args.dt
        self.obs_len = args.obs_len
        self.pred_len = args.pred_len

        self.alpha = 0.25
        self.gamma = 2
        self.eps=1e-9
        self.sg_idx = np.array(range(self.pred_len))
        self.sg_idx = np.flip(self.pred_len-1-self.sg_idx[::(self.pred_len//args.num_goal)])


        self.max_iter = int(args.max_iter)
        self.lr = args.lr


        self.ckpt_dir = os.path.join(args.ckpt_dir, self.name)
        self.ckpt_load_iter = args.ckpt_load_iter
        mkdirs(self.ckpt_dir)

        num_filters = [32, 32, 64, 64, 64, 128]
        # input = env + past trajectories + lg / output = env + sg(including lg)
        self.sg_unet = Unet(input_channels=3, num_classes=len(self.sg_idx), num_filters=num_filters,
                            apply_last_layer=True, padding=True).to(self.device)

        self.optim = optim.Adam(
            list(self.sg_unet.parameters()),
            lr=self.lr
        )
        print('Start loading data...')
        if self.dataset_name == 'nuScenes':
            cfg = Config('nuscenes_train', False, create_dirs=True)
            torch.set_default_dtype(torch.float32)
            log = open('log.txt', 'a+')
            self.train_loader = data_generator(cfg, log, split='train', phase='training',
                                               batch_size=args.batch_size, device=self.device, scale=args.scale, shuffle=True)
            cfg = Config('nuscenes', False, create_dirs=True)
            torch.set_default_dtype(torch.float32)
            log = open('log.txt', 'a+')
            self.val_loader = data_generator(cfg, log, split='val', phase='testing',
                                             batch_size=args.batch_size, device=self.device, scale=args.scale, shuffle=True)
            print(
                'There are {} iterations per epoch'.format(len(self.train_loader.idx_list))
            )
        else:
            _, self.train_loader = data_loader(self.args, 'train', shuffle=True)
            _, self.val_loader = data_loader(self.args, 'val', shuffle=True)
            print(
                'There are {} iterations per epoch'.format(len(self.train_loader.dataset) / args.batch_size)
            )
        print('...done')

        hg = heatmap_generation(args.dataset_name, self.obs_len, args.heatmap_size, sg_idx=self.sg_idx, device=self.device)
        self.make_heatmap = hg.make_heatmap



    def train(self):
        self.set_mode(train=True)
        data_loader = self.train_loader
        if self.dataset_name == 'nuScenes':
            iter_per_epoch = len(data_loader.idx_list)
        else:
            iterator = iter(data_loader)
            iter_per_epoch = len(iterator)
        start_iter = 1
        epoch = int(start_iter / iter_per_epoch)

        for iteration in range(start_iter, self.max_iter + 1):
            # reset data iterators for each epoch
            if iteration % iter_per_epoch == 0:
                print('==== epoch %d done ====' % epoch)
                epoch +=1
                if self.dataset_name == 'nuScenes':
                    data_loader.is_epoch_end()
                else:
                    iterator = iter(data_loader)

            if self.dataset_name == 'nuScenes':
                data = data_loader.next_sample()
                if data is None:
                    continue
            else:
                data = next(iterator)

            (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
             map_info, inv_h_t,
             local_map, local_ic, local_homo) = data

            batch_size = obs_traj.size(1)

            obs_heat_map, sg_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map, aug=True)

            recon_sg_heat = self.sg_unet.forward(torch.cat([obs_heat_map, lg_heat_map], dim=1))
            recon_sg_heat = F.sigmoid(recon_sg_heat)
            normalized_recon_sg_heat = []
            for i in range(len(self.sg_idx)):
                sg_map = recon_sg_heat[:,i]
                normalized_recon_sg_heat.append(F.normalize(sg_map.view((sg_map.shape[0], -1)), p=1))
            recon_sg_heat = torch.stack(normalized_recon_sg_heat, dim=1)
            sg_heat_map= sg_heat_map.view(sg_heat_map.shape[0], len(self.sg_idx), -1)

            sg_recon_loss = - (
            self.alpha * sg_heat_map * torch.log(recon_sg_heat + self.eps) * ((1 - recon_sg_heat) ** self.gamma) \
            + (1 - self.alpha) * (1 - sg_heat_map) * torch.log(1 - recon_sg_heat + self.eps) * (
                recon_sg_heat ** self.gamma)).sum().div(batch_size)
            loss = sg_recon_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # save model parameters
            if iteration % (iter_per_epoch*10) == 0:
                self.save_checkpoint(iteration)

        # save model parameters
        self.save_checkpoint(self.max_iter)


    def set_mode(self, train=True):
        if train:
            self.sg_unet.train()
        else:
            self.sg_unet.eval()


    def save_checkpoint(self, iteration):
        path = os.path.join(
            self.ckpt_dir,
            'iter_%s_%s.pt' % (iteration, self.model_name))
        torch.save(self.sg_unet, path)

    def load_checkpoint(self):
        path = os.path.join(
            self.ckpt_dir,
            'iter_%s_%s.pt' % (self.ckpt_load_iter, self.model_name)
        )
        if self.device == 'cuda':
            self.sg_unet = torch.load(path)

        else:
            self.sg_unet = torch.load(path, map_location='cpu')
