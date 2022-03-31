from torch.distributions import kl_divergence
import torch.optim as optim
from util import *
from model import *
from data.loader import data_loader
from data.nuscenes.config import Config
from data.nuscenes_dataloader import data_generator
import numpy as np

class Solver(object):

    ####
    def __init__(self, args):

        self.args = args
        self.name = '%s_%s_z_%s_enc_hD_%s_dec_hD_%s_mlpD_%s_map_featD_%s_map_mlpD_%s_lr_%s_klw_%s_fb_%s_scale_%s_n_goal_%s_run_%s' % \
                    (args.dataset_name, args.model_name, args.z_dim, args.encoder_h_dim,
                     args.decoder_h_dim, args.mlp_dim, args.map_feat_dim , args.map_mlp_dim,
                     args.lr, args.kl_weight, args.fb, args.scale, args.num_goal, args.run_id)

        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.fb = args.fb
        self.anneal_epoch = args.anneal_epoch
        self.device = args.device
        self.dt=args.dt
        self.obs_len = args.obs_len
        self.pred_len = args.pred_len

        self.alpha = 0.25
        self.gamma = 2
        self.eps=1e-9
        self.sg_idx = np.array(range(self.pred_len))
        self.sg_idx = np.flip(self.pred_len-1-self.sg_idx[::(self.pred_len//args.num_goal)])
        self.z_dim = args.z_dim
        self.kl_weight=args.kl_weight

        self.max_iter = int(args.max_iter)
        self.scale = args.scale
        self.lr = args.lr

        self.ckpt_dir = os.path.join(args.ckpt_dir, self.name)
        self.ckpt_load_iter = args.ckpt_load_iter
        mkdirs(self.ckpt_dir)

        if self.device == 'cuda':
            self.lg_cvae = torch.load(args.pretrained_lg_path)
        else:
            self.lg_cvae = torch.load(args.pretrained_lg_path, map_location='cpu')

        self.encoderMx = EncoderX(
            args.z_dim,
            enc_h_dim=args.encoder_h_dim,
            mlp_dim=args.mlp_dim,
            map_mlp_dim=args.map_mlp_dim,
            map_feat_dim=args.map_feat_dim,
            map_h_dim=np.prod(np.array(self.lg_cvae.unet_enc_feat.shape[1:])),
            device=self.device).to(self.device)
        self.encoderMy = EncoderY(
            args.z_dim,
            enc_h_dim=args.encoder_h_dim,
            mlp_dim=args.mlp_dim,
            device=self.device).to(self.device)
        self.decoderMy = Decoder(
            args.pred_len,
            dec_h_dim=args.decoder_h_dim,
            enc_h_dim=args.encoder_h_dim,
            mlp_dim=args.mlp_dim,
            z_dim=args.z_dim,
            device=args.device).to(self.device)



        # get VAE parameters
        params = \
            list(self.encoderMx.parameters()) + \
            list(self.encoderMy.parameters()) + \
            list(self.decoderMy.parameters())
        # create optimizers
        self.optim_vae = optim.Adam(
            params,
            lr=self.lr,
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
            self.val_loader = data_generator(cfg, log, split='test', phase='testing',
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

        hg = heatmap_generation(args.dataset_name, self.obs_len, args.heatmap_size, sg_idx=None, device=self.device)
        self.make_heatmap = hg.make_heatmap




    ####
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

            obs_heat_map, _ =  self.make_heatmap(local_ic, local_map, aug=True)

            #-------- map encoding from lgvae --------
            unet_enc_feat = self.lg_cvae.unet.down_forward(obs_heat_map)

            #-------- trajectories --------
            (hx, mux, log_varx) \
                = self.encoderMx(obs_traj_st, seq_start_end, unet_enc_feat, local_homo, train=True)
            (muy, log_vary) \
                = self.encoderMy(obs_traj_st[-1], fut_vel_st, seq_start_end, hx, train=True)

            p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))
            q_dist = Normal(muy, torch.sqrt(torch.exp(log_vary)))


            # TF, z~posterior
            fut_rel_pos_dist_tf_post = self.decoderMy(
                obs_traj_st[-1],
                obs_traj[-1, :, :2],
                hx,
                q_dist.rsample(),
                fut_traj[list(self.sg_idx), :, :2].permute(1,0,2), # goal
                self.sg_idx,
                fut_vel_st # TF
            )


            # NO TF, z~prior
            fut_rel_pos_dist_prior = self.decoderMy(
                obs_traj_st[-1],
                obs_traj[-1, :, :2],
                hx,
                p_dist.rsample(),
                fut_traj[list(self.sg_idx), :, :2].permute(1, 0, 2),  # goal
                self.sg_idx,
            )


            ll_tf_post = fut_rel_pos_dist_tf_post.log_prob(fut_vel_st).sum().div(batch_size)
            ll_prior = fut_rel_pos_dist_prior.log_prob(fut_vel_st).sum().div(batch_size)

            loss_kl = kl_divergence(q_dist, p_dist)
            loss_kl = torch.clamp(loss_kl, min=self.fb).sum().div(batch_size)

            loglikelihood= ll_tf_post + ll_prior
            traj_elbo = loglikelihood - self.kl_weight * loss_kl

            loss = - traj_elbo

            self.optim_vae.zero_grad()
            loss.backward()
            self.optim_vae.step()

            # save model parameters
            if iteration % (iter_per_epoch*10) == 0:
                self.save_checkpoint(iteration)
        # save model parameters
        self.save_checkpoint(self.max_iter)


    def set_mode(self, train=True):
        if train:
            self.encoderMx.train()
            self.encoderMy.train()
            self.decoderMy.train()
        else:
            self.encoderMx.eval()
            self.encoderMy.eval()
            self.decoderMy.eval()


    def save_checkpoint(self, iteration):

        encoderMx_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderMx.pt' % iteration
        )
        encoderMy_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderMy.pt' % iteration
        )
        decoderMy_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderMy.pt' % iteration
        )

        torch.save(self.encoderMx, encoderMx_path)
        torch.save(self.encoderMy, encoderMy_path)
        torch.save(self.decoderMy, decoderMy_path)
    ####
    def load_checkpoint(self):

        encoderMx_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderMx.pt' % self.ckpt_load_iter
        )
        encoderMy_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderMy.pt' % self.ckpt_load_iter
        )
        decoderMy_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderMy.pt' % self.ckpt_load_iter
        )

        if self.device == 'cuda':
            self.encoderMx = torch.load(encoderMx_path)
            self.encoderMy = torch.load(encoderMy_path)
            self.decoderMy = torch.load(decoderMy_path)
        else:
            self.encoderMx = torch.load(encoderMx_path, map_location='cpu')
            self.encoderMy = torch.load(encoderMy_path, map_location='cpu')
            self.decoderMy = torch.load(decoderMy_path, map_location='cpu')