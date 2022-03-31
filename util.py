from scipy import ndimage
import numpy as np
import torch
from torchvision import transforms
import os
import argparse
import cv2

class heatmap_generation(object):
    def __init__(self, dataset, obs_len, heatmap_size, sg_idx=None, device='cpu'):
        self.obs_len = obs_len
        self.device = device
        self.sg_idx = sg_idx
        self.heatmap_size = heatmap_size
        if dataset == 'pfsd':
            self.make_heatmap = self.create_psfd_heatmap
        elif dataset == 'sdd':
            self.make_heatmap = self.create_sdd_heatmap
        else:
            self.make_heatmap = self.create_nu_heatmap

    def create_psfd_heatmap(self, local_ic, local_map, aug=False):
        heatmaps = []
        for i in range(len(local_ic)):
            all_heatmap = [local_map[i]]
            heatmap = np.zeros((self.heatmap_size, self.heatmap_size))
            for t in range(self.obs_len):
                heatmap[local_ic[i, t, 0], local_ic[i, t, 1]] = 1
                # as Y-net used variance 4 for the GT heatmap representation.
            heatmap = ndimage.filters.gaussian_filter(heatmap, sigma=2)
            all_heatmap.append(heatmap / heatmap.sum())

            if self.sg_idx is None:
                heatmap = np.zeros((self.heatmap_size, self.heatmap_size))
                heatmap[local_ic[i, -1, 0], local_ic[i, -1, 1]] = 1
                heatmap = ndimage.filters.gaussian_filter(heatmap, sigma=2)
                all_heatmap.append(heatmap)
            else:
                for t in (self.sg_idx + self.obs_len):
                    heatmap = np.zeros((self.heatmap_size, self.heatmap_size))
                    heatmap[local_ic[i, t, 0], local_ic[i, t, 1]] = 1
                    heatmap = ndimage.filters.gaussian_filter(heatmap, sigma=2)
                    all_heatmap.append(heatmap)
            heatmaps.append(np.stack(all_heatmap))

        heatmaps = torch.tensor(np.stack(heatmaps)).float().to(self.device)
        if aug:
            degree = np.random.choice([0, 90, 180, -90])
            heatmaps = transforms.Compose([
                transforms.RandomRotation(degrees=(degree, degree))
            ])(heatmaps)

        if self.sg_idx is None:
            return heatmaps[:, :2], heatmaps[:, 2:]
        else:
            return heatmaps[:, :2], heatmaps[:, 2:], heatmaps[:, -1].unsqueeze(1)


    def make_one_heatmap(self, local_map, local_ic):
        map_size = local_map.shape[0]
        half = self.heatmap_size // 2
        if map_size < self.heatmap_size:
            heatmap = np.zeros_like(local_map)
            heatmap[local_ic[0], local_ic[1]] = 1
            heatmap = ndimage.filters.gaussian_filter(heatmap, sigma=2)
            extended_map = np.zeros((self.heatmap_size, self.heatmap_size))
            extended_map[half - map_size // 2:half + map_size // 2,half - map_size // 2:half + map_size // 2] = heatmap
            heatmap = extended_map
        else:
            heatmap = np.zeros_like(local_map)
            heatmap[local_ic[0], local_ic[1]] = 1000
            if map_size > 1000:
                heatmap = cv2.resize(ndimage.filters.gaussian_filter(heatmap, sigma=2),
                                           dsize=((map_size + self.heatmap_size) // 2, (map_size + self.heatmap_size) // 2))
            heatmap = cv2.resize(ndimage.filters.gaussian_filter(heatmap, sigma=2),
                                       dsize=(self.heatmap_size, self.heatmap_size))
            heatmap = heatmap / heatmap.sum()
            heatmap = ndimage.filters.gaussian_filter(heatmap, sigma=2)
        return heatmap


    def create_sdd_heatmap(self, local_ic, local_map, aug=False):
        heatmaps=[]
        half = self.heatmap_size//2
        for i in range(len(local_ic)):
            map_size = local_map[i].shape[0]
            # past
            if map_size < self.heatmap_size:
                env = np.full((self.heatmap_size,self.heatmap_size),3)
                env[half-map_size//2:half+map_size//2, half-map_size//2:half+map_size//2] = local_map[i]
                all_heatmap = [env/5]
                heatmap = np.zeros_like(local_map[i])
                heatmap[local_ic[i, :self.obs_len, 0], local_ic[i, :self.obs_len, 1]] = 1
                heatmap= ndimage.filters.gaussian_filter(heatmap, sigma=2)
                heatmap = heatmap / heatmap.sum()
                extended_map = np.zeros((self.heatmap_size, self.heatmap_size))
                extended_map[half-map_size//2:half+map_size//2, half-map_size//2:half+map_size//2] = heatmap
                all_heatmap.append(extended_map)
            else:
                env = cv2.resize(local_map[i], dsize=(self.heatmap_size, self.heatmap_size))
                all_heatmap = [env/5]
                heatmap = np.zeros_like(local_map[i])
                heatmap[local_ic[i, :self.obs_len, 0], local_ic[i, :self.obs_len, 1]] = 100
                if map_size > 1000:
                    heatmap = cv2.resize(ndimage.filters.gaussian_filter(heatmap, sigma=2),
                                               dsize=((map_size+self.heatmap_size)//2, (map_size+self.heatmap_size)//2))
                    heatmap = heatmap / heatmap.sum()
                heatmap = cv2.resize(ndimage.filters.gaussian_filter(heatmap, sigma=2), dsize=(self.heatmap_size, self.heatmap_size))
                if map_size > 3500:
                    heatmap[np.where(heatmap > 0)] = 1
                else:
                    heatmap = heatmap / heatmap.sum()
                heatmap = ndimage.filters.gaussian_filter(heatmap, sigma=2)
                all_heatmap.append(heatmap / heatmap.sum())

            # future
            if self.sg_idx is None:
                heatmap = self.make_one_heatmap(local_map[i], local_ic[i, -1])
                all_heatmap.append(heatmap)
            else:
                for j in (self.sg_idx + self.obs_len):
                    heatmap = self.make_one_heatmap(local_map[i], local_ic[i, j])
                    all_heatmap.append(heatmap)
            heatmaps.append(np.stack(all_heatmap))

        heatmaps = torch.tensor(np.stack(heatmaps)).float().to(self.device)

        if aug:
            degree = np.random.choice([0,90,180, -90])
            heatmaps = transforms.Compose([
                transforms.RandomRotation(degrees=(degree, degree))
            ])(heatmaps)
        if self.sg_idx is None:
            return heatmaps[:, :2], heatmaps[:, 2:]
        else:
            return heatmaps[:, :2], heatmaps[:, 2:], heatmaps[:, -1].unsqueeze(1)





    def create_nu_heatmap(self, local_ic, local_map, aug=False):
        heatmaps=[]
        half = self.heatmap_size//2
        for i in range(len(local_ic)):
            map_size = local_map[i].shape[0]
            # past
            if map_size < self.heatmap_size:
                env = np.full((self.heatmap_size,self.heatmap_size),1)
                env[half-map_size//2:half+map_size//2, half-map_size//2:half+map_size//2] = local_map[i]
                all_heatmap = [env]
                heatmap = np.zeros_like(local_map[i])
                heatmap[local_ic[i, :self.obs_len, 0], local_ic[i, :self.obs_len, 1]] = 1
                heatmap= ndimage.filters.gaussian_filter(heatmap, sigma=2)
                heatmap = heatmap / heatmap.sum()
                extended_map = np.zeros((self.heatmap_size, self.heatmap_size))
                extended_map[half-map_size//2:half+map_size//2, half-map_size//2:half+map_size//2] = heatmap
                all_heatmap.append(extended_map)
            else:
                env = cv2.resize(local_map[i], dsize=(self.heatmap_size, self.heatmap_size))
                all_heatmap = [env]
                heatmap = np.zeros_like(local_map[i])
                heatmap[local_ic[i, :self.obs_len, 0], local_ic[i, :self.obs_len, 1]] = 100
                if map_size > 1000:
                    heatmap = cv2.resize(ndimage.filters.gaussian_filter(heatmap, sigma=2),
                                               dsize=((map_size+self.heatmap_size)//2, (map_size+self.heatmap_size)//2))
                    heatmap = heatmap / heatmap.sum()
                heatmap = cv2.resize(ndimage.filters.gaussian_filter(heatmap, sigma=2), dsize=(self.heatmap_size, self.heatmap_size))
                if map_size > 3500:
                    heatmap[np.where(heatmap > 0)] = 1
                else:
                    heatmap = heatmap / heatmap.sum()
                heatmap = ndimage.filters.gaussian_filter(heatmap, sigma=2)
                all_heatmap.append(heatmap / heatmap.sum())

            # future
            if self.sg_idx is None:
                heatmap = self.make_one_heatmap(local_map[i], local_ic[i, -1])
                all_heatmap.append(heatmap)
            else:
                for j in (self.sg_idx + self.obs_len):
                    heatmap = self.make_one_heatmap(local_map[i], local_ic[i, j])
                    all_heatmap.append(heatmap)
            heatmaps.append(np.stack(all_heatmap))

        heatmaps = torch.tensor(np.stack(heatmaps)).float().to(self.device)

        if aug:
            degree = np.random.choice([0,90,180, -90])
            heatmaps = transforms.Compose([
                transforms.RandomRotation(degrees=(degree, degree))
            ])(heatmaps)
        if self.sg_idx is None:
            return heatmaps[:, :2], heatmaps[:, 2:]
        else:
            return heatmaps[:, :2], heatmaps[:, 2:], heatmaps[:, -1].unsqueeze(1)




def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def derivative_of(x, dt=1):

    if x[~np.isnan(x)].shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[~np.isnan(x)] = np.gradient(x[~np.isnan(x)], dt)
    return dx

def integrate_samples(v, p_0, dt=1):
    """
    Integrates deterministic samples of velocity.

    :param v: Velocity samples
    :return: Position samples
    """
    v=v.permute(1, 0, 2)
    abs_traj = torch.cumsum(v, dim=1) * dt + p_0.unsqueeze(1)
    return  abs_traj.permute((1, 0, 2))





def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    https://github.com/agrimgupta92/sgan
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    # loss = pred_traj_gt - pred_traj
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    https://github.com/agrimgupta92/sgan
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)
