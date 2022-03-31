import logging
import os
import math
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from util import derivative_of

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import imageio
from skimage.transform import resize
import pickle5

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list,
     map_path, inv_h_t,
     local_map, local_ic, local_homo, scale) = zip(*data)
    scale = scale[0]

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj = torch.stack(obs_seq_list, dim=0).permute(2, 0, 1)
    fut_traj = torch.stack(pred_seq_list, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)

    inv_h_t = np.stack(inv_h_t)
    local_ic = np.stack(local_ic)
    local_homo = torch.tensor(np.stack(local_homo)).float().to(obs_traj.device)

    obs_traj_st = obs_traj.clone()
    obs_traj_st[:, :, :2] = (obs_traj_st[:,:,:2] - obs_traj_st[-1, :, :2]) / scale
    obs_traj_st[:, :, 2:] /= scale
    out = [
        obs_traj, fut_traj, obs_traj_st, fut_traj[:,:,2:4] / scale, seq_start_end,
        map_path, inv_h_t,
        local_map, local_ic, local_homo
    ]

    return tuple(out)



def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)




def transform(image, resize):
    im = Image.fromarray(image[0])

    image = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])(im)
    return image


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, data_split, device='cpu', scale=100
    ):

        super(TrajectoryDataset, self).__init__()

        self.obs_len = 8
        self.pred_len = 12
        self.skip = 1
        self.scale = scale
        self.seq_len = self.obs_len + self.pred_len
        self.delim = ' '
        self.device = device
        if data_split == 'val':
            data_split = 'test'
        self.map_dir = os.path.join(data_dir, 'SDD_semantic_maps', data_split + '_masks')
        self.data_path = os.path.join(data_dir, 'sdd_' + data_split + '.pkl')
        dt=0.4
        min_ped=0

        self.seq_len = self.obs_len + self.pred_len


        n_state = 6
        num_peds_in_seq = []
        seq_list = []

        obs_frame_num = []
        fut_frame_num = []
        scene_names = []
        local_map_size=[]

        self.stats={}
        self.maps={}
        for file in os.listdir(self.map_dir):
            m = imageio.imread(os.path.join(self.map_dir, file)).astype(float)
            self.maps.update({file.split('.')[0]:m})


        with open(self.data_path, 'rb') as f:
            data = pickle5.load(f)

        data = pd.DataFrame(data)
        scenes = data['sceneId'].unique()
        for s in scenes:
            # incomplete dataset - trajectories are not aligned with segmentation.
            if ('nexus_2' in s) or ('hyang_4' in s):
                continue
            scene_data = data[data['sceneId'] == s]
            scene_data = scene_data.sort_values(by=['frame', 'trackId'], inplace=False)


            frames = scene_data['frame'].unique().tolist()
            scene_data = np.array(scene_data)
            map_size = self.maps[s + '_mask'].shape
            scene_data[:,2] = np.clip(scene_data[:,2], a_min=None, a_max=map_size[1]-1)
            scene_data[:,3] = np.clip(scene_data[:,3], a_min=None, a_max=map_size[0]-1)


            frame_data = []
            for frame in frames:
                frame_data.append(scene_data[scene_data[:, 0]==frame])

            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip))

            this_scene_seq = []

            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len],
                    axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # unique agent id

                curr_seq = np.zeros((len(peds_in_curr_seq), n_state, self.seq_len))
                num_peds_considered = 0
                ped_ids = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if (pad_end - pad_front != self.seq_len) or (curr_ped_seq.shape[0] != self.seq_len):
                        continue
                    ped_ids.append(ped_id)
                    # x,y,x',y',x'',y''
                    x = curr_ped_seq[:, 2].astype(float)
                    y = curr_ped_seq[:, 3].astype(float)
                    vx = derivative_of(x, dt)
                    vy = derivative_of(y, dt)
                    ax = derivative_of(vx, dt)
                    ay = derivative_of(vy, dt)

                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = np.stack([x, y, vx, vy, ax, ay])
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    num_peds_in_seq.append(num_peds_considered)
                    seq_list.append(curr_seq[:num_peds_considered])
                    this_scene_seq.append(curr_seq[:num_peds_considered, :2])
                    obs_frame_num.append(np.ones((num_peds_considered, self.obs_len)) * frames[idx:idx + self.obs_len])
                    fut_frame_num.append(
                        np.ones((num_peds_considered, self.pred_len)) * frames[idx + self.obs_len:idx + self.seq_len])
                    scene_names.append([s] * num_peds_considered)


            this_scene_seq = np.concatenate(this_scene_seq)

            per_step_dist = []
            for traj in this_scene_seq:
                traj = traj.transpose(1, 0)
                per_step_dist.append(np.sqrt(((traj[1:] - traj[:-1]) ** 2).sum(1)).sum())
            per_step_dist = np.array(per_step_dist)

            per_step_dist = np.clip(per_step_dist, a_min=240, a_max=None)

            local_map_size.extend(np.round(per_step_dist).astype(int))
            # print( self.maps[s + '_mask'].shape, ': ' ,(per_step_dist).max())

        seq_list = np.concatenate(seq_list, axis=0) # (32686, 2, 16)
        self.obs_frame_num = np.concatenate(obs_frame_num, axis=0)
        self.fut_frame_num = np.concatenate(fut_frame_num, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.map_file_name = np.concatenate(scene_names)
        self.num_seq = len(self.obs_traj)
        self.local_map_size = np.stack(local_map_size)
        self.local_ic = [[]] * self.num_seq
        self.local_homo = [[]] * self.num_seq

        print(self.seq_start_end[-1])

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        global_map = self.maps[self.map_file_name[index] + '_mask']
        inv_h_t = np.expand_dims(np.eye(3), axis=0)
        all_traj = torch.cat([self.obs_traj[index, :2, :], self.pred_traj[index, :2, :]],
                             dim=1).detach().cpu().numpy().transpose((1, 0))
        if len(self.local_ic[index]) == 0:
            local_map, local_ic, local_homo = self.get_local_map_ic(global_map, all_traj, zoom=1,
                                                                    radius=self.local_map_size[index],
                                                                    compute_local_homo=True)
            self.local_ic[index] = local_ic
            self.local_homo[index] = local_homo
        else:
            local_map, _, _ = self.get_local_map_ic(global_map, all_traj, zoom=1, radius=self.local_map_size[index])
            local_ic = self.local_ic[index]
            local_homo = self.local_homo[index]

        out = [
            self.obs_traj[index].to(self.device), self.pred_traj[index].to(self.device),
            self.map_file_name[index] + '_mask', inv_h_t,
            local_map, local_ic, local_homo, self.scale
        ]
        return out


    def get_local_map_ic(self, global_map, all_traj, zoom=10, radius=8, compute_local_homo=False):
            radius = radius * zoom
            context_size = radius * 2
            expanded_obs_img = np.full((global_map.shape[0] + context_size, global_map.shape[1] + context_size),
                                       3, dtype=np.float32)
            expanded_obs_img[radius:-radius, radius:-radius] = global_map.astype(np.float32)  # 99~-99

            all_pixel = all_traj[:,[1,0]]
            all_pixel = radius + np.round(all_pixel).astype(int)

            local_map = expanded_obs_img[all_pixel[7, 0] - radius: all_pixel[7, 0] + radius,
                        all_pixel[7, 1] - radius: all_pixel[7, 1] + radius]

            all_pixel_local = None
            h = None
            if compute_local_homo:
                fake_pt = [all_traj[7]]
                per_pixel_dist = radius // 10

                for i in range(per_pixel_dist, radius // 2 - per_pixel_dist, per_pixel_dist):
                    fake_pt.append(all_traj[7] + [i, i] + np.random.rand(2) * (per_pixel_dist//2))
                    fake_pt.append(all_traj[7] + [-i, -i] + np.random.rand(2) * (per_pixel_dist//2))
                    fake_pt.append(all_traj[7] + [i, -i] + np.random.rand(2) * (per_pixel_dist//2))
                    fake_pt.append(all_traj[7] + [-i, i] + np.random.rand(2) * (per_pixel_dist//2))
                fake_pt = np.array(fake_pt)


                fake_pixel = fake_pt[:,[1,0]]
                fake_pixel = radius + np.round(fake_pixel).astype(int)

                temp_map_val = []
                for i in range(len(fake_pixel)):
                    temp_map_val.append(expanded_obs_img[fake_pixel[i, 0], fake_pixel[i, 1]])
                    expanded_obs_img[fake_pixel[i, 0], fake_pixel[i, 1]] = i + 10

                fake_local_pixel = []
                for i in range(len(fake_pixel)):
                    fake_local_pixel.append([np.where(local_map == i + 10)[0][0], np.where(local_map == i + 10)[1][0]])
                    expanded_obs_img[fake_pixel[i, 0], fake_pixel[i, 1]] = temp_map_val[i]

                h, _ = cv2.findHomography(np.array([fake_local_pixel]), np.array(fake_pt))

                all_pixel_local = np.matmul(np.concatenate([all_traj, np.ones((len(all_traj), 1))], axis=1),
                                            np.linalg.pinv(np.transpose(h)))
                all_pixel_local /= np.expand_dims(all_pixel_local[:, 2], 1)
                all_pixel_local = np.round(all_pixel_local).astype(int)[:, :2]

            return local_map, all_pixel_local, h
