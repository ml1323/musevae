from model import *
import torch.nn.functional as nnf

import matplotlib.pyplot as plt
from util import *
import numpy as np
import torch.nn.functional as F
from data.loader import data_loader
from data.nuscenes.config import Config
from data.nuscenes_dataloader import data_generator


class Solver(object):
    def __init__(self, args):

        self.args = args
        self.device = args.device
        self.dt=args.dt
        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.heatmap_size = args.heatmap_size
        self.dataset_name = args.dataset_name
        self.scale = args.scale
        self.n_w = args.n_w
        self.n_z = args.n_z

        self.eps=1e-9
        self.sg_idx = np.array(range(self.pred_len))
        self.sg_idx = np.flip(self.pred_len-1-self.sg_idx[::(self.pred_len//args.num_goal)])

        self.ckpt_dir = os.path.join(args.ckpt_dir, "pretrained_models_" + args.dataset_name)

        if args.dataset_name == 'nuScenes':
            cfg = Config('nuscenes', False, create_dirs=True)
            torch.set_default_dtype(torch.float32)
            log = open('log.txt', 'a+')
            self.data_loader = data_generator(cfg, log, split='test', phase='testing',
                                         batch_size=args.batch_size, device=args.device, scale=args.scale,
                                         shuffle=False)
        else:
            _, self.data_loader = data_loader(self.args, 'test', shuffle=False)

        hg = heatmap_generation(args.dataset_name, self.obs_len, args.heatmap_size, sg_idx=None, device=self.device)
        self.make_heatmap = hg.make_heatmap
        self.make_one_heatmap = hg.make_one_heatmap


    def all_evaluation(self):
        self.set_mode(train=False)
        all_ade =[]
        all_fde =[]
        with torch.no_grad():
            if self.dataset_name == 'nuScenes':
                while not self.data_loader.is_epoch_end():
                    batch = self.data_loader.next_sample()
                    if batch is None:
                        continue
                    ade, fde = self.compute(batch)
                    all_ade.append(ade)
                    all_fde.append(fde)
            else:
                for batch in self.data_loader:
                    ade, fde = self.compute(batch)
                    all_ade.append(ade)
                    all_fde.append(fde)

            all_ade=torch.cat(all_ade, dim=1).cpu().numpy()
            all_fde=torch.cat(all_fde, dim=1).cpu().numpy()

            ade_min = np.min(all_ade, axis=0).mean()/self.pred_len
            fde_min = np.min(all_fde, axis=0).mean()
        return ade_min, fde_min


    def compute(self, batch):
        (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
         map_info, inv_h_t,
         local_map, local_ic, local_homo) = batch
        batch_size = obs_traj.size(1)

        obs_heat_map, _ = self.make_heatmap(local_ic, local_map)

        self.lg_cvae.forward(obs_heat_map, None, training=False)
        fut_rel_pos_dists = []
        pred_lg_wcs = []
        pred_sg_wcs = []

        ####### long term goals and the corresponding (deterministic) short term goals ########
        w_priors = []
        for _ in range(self.n_w):
            w_priors.append(self.lg_cvae.prior_latent_space.sample())

        for w_prior in w_priors:
            # -------- long term goal --------
            pred_lg_heat = F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, w_prior))

            pred_lg_wc = []
            pred_lg_ics = []
            for i in range(batch_size):
                map_size = local_map[i].shape
                pred_lg_ic = []
                if self.dataset_name == 'pfsd':
                    for heat_map in pred_lg_heat[i]:
                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_lg_ic.append(argmax_idx)
                else:
                    for heat_map in pred_lg_heat[i]:
                        heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                   size=map_size, mode='bicubic',
                                                   align_corners=False).squeeze(0).squeeze(0)
                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_lg_ic.append(argmax_idx)

                pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)
                pred_lg_ics.append(pred_lg_ic)
                back_wc = torch.matmul(
                    torch.cat([pred_lg_ic, torch.ones((len(pred_lg_ic), 1)).to(self.device)], dim=1),
                    torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                pred_lg_wc.append(back_wc[0, :2] / back_wc[0, 2])
            pred_lg_wc = torch.stack(pred_lg_wc)
            pred_lg_wcs.append(pred_lg_wc)

            # -------- short term goal --------
            pred_lg_heat_from_ic = []
            if self.dataset_name == 'pfsd':
                for coord in pred_lg_ics:
                    heat_map_traj = np.zeros((self.heatmap_size, self.heatmap_size))
                    heat_map_traj[int(coord[0, 0]), int(coord[0, 1])] = 1
                    heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                    pred_lg_heat_from_ic.append(heat_map_traj)
            else:
                for i in range(len(pred_lg_ics)):
                    pred_lg_heat_from_ic.append(self.make_one_heatmap(local_map[i], pred_lg_ics[i][
                        0].detach().cpu().numpy().astype(int)))
            pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                self.device)
            pred_sg_heat = F.sigmoid(
                self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))

            pred_sg_wc = []
            for i in range(batch_size):
                pred_sg_ic = []
                if self.dataset_name == 'pfsd':
                    for heat_map in pred_sg_heat[i]:
                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_sg_ic.append(argmax_idx)
                else:
                    map_size = local_map[i].shape
                    for heat_map in pred_sg_heat[i]:
                        heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                   size=map_size, mode='bicubic',
                                                   align_corners=False).squeeze(0).squeeze(0)
                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_sg_ic.append(argmax_idx)

                pred_sg_ic = torch.tensor(pred_sg_ic).float().to(self.device)

                back_wc = torch.matmul(
                    torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(self.device)], dim=1),
                    torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                back_wc /= back_wc[:, 2].unsqueeze(1)
                pred_sg_wc.append(back_wc[:, :2])
            pred_sg_wc = torch.stack(pred_sg_wc)
            pred_sg_wcs.append(pred_sg_wc)

        ##### trajectories per long&short goal ####
        # -------- Micro --------
        (hx, mux, log_varx) \
            = self.encoderMx(obs_traj_st, seq_start_end, self.lg_cvae.unet_enc_feat, local_homo)

        p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))
        z_priors = []
        for _ in range(self.n_z):
            z_priors.append(p_dist.sample())

        for pred_sg_wc in pred_sg_wcs:
            for z_prior in z_priors:
                fut_rel_pos_dist_prior = self.decoderMy(
                    obs_traj_st[-1],
                    obs_traj[-1, :, :2],
                    hx,
                    z_prior,
                    pred_sg_wc,  # goal prediction
                    self.sg_idx
                )
                fut_rel_pos_dists.append(fut_rel_pos_dist_prior)

        ade, fde = [], []
        for dist in fut_rel_pos_dists:
            pred_fut_traj = integrate_samples(dist.rsample() * self.scale, obs_traj[-1, :, :2], dt=self.dt)
            ade.append(displacement_error(
                pred_fut_traj, fut_traj[:, :, :2], mode='raw'
            ))
            fde.append(final_displacement_error(
                pred_fut_traj[-1], fut_traj[-1, :, :2], mode='raw'
            ))
        ade = torch.stack(ade)
        fde = torch.stack(fde)
        return ade, fde


    def load_checkpoint(self):
        sg_unet_path = os.path.join(
            self.ckpt_dir,
            'sg_net.pt'
        )
        encoderMx_path = os.path.join(
            self.ckpt_dir,
            'encoderMx.pt'
        )
        encoderMy_path = os.path.join(
            self.ckpt_dir,
            'encoderMy.pt'
        )
        decoderMy_path = os.path.join(
            self.ckpt_dir,
            'decoderMy.pt'
        )
        lg_cvae_path = os.path.join(
            self.ckpt_dir,
            'lg_cvae.pt'
        )
        if self.device == 'cuda':
            self.encoderMx = torch.load(encoderMx_path)
            self.encoderMy = torch.load(encoderMy_path)
            self.decoderMy = torch.load(decoderMy_path)
            self.lg_cvae = torch.load(lg_cvae_path)
            self.sg_unet = torch.load(sg_unet_path)
        else:
            self.encoderMx = torch.load(encoderMx_path, map_location='cpu')
            self.encoderMy = torch.load(encoderMy_path, map_location='cpu')
            self.decoderMy = torch.load(decoderMy_path, map_location='cpu')
            self.lg_cvae = torch.load(lg_cvae_path, map_location='cpu')
            self.sg_unet = torch.load(sg_unet_path, map_location='cpu')
        print('ckpt loaded from ', self.ckpt_dir)


    def set_mode(self, train=True):
        if train:
            self.sg_unet.train()
            self.lg_cvae.train()
            self.encoderMx.train()
            self.encoderMy.train()
            self.decoderMy.train()
        else:
            self.sg_unet.eval()
            self.lg_cvae.eval()
            self.encoderMx.eval()
            self.encoderMy.eval()
            self.decoderMy.eval()
