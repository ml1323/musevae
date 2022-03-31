import logging
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
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

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.stack(obs_seq_list).permute(2, 0, 1)
    fut_traj = torch.stack(pred_seq_list).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)

    map_path = np.stack(map_path)
    inv_h_t = np.stack(inv_h_t)
    local_map = np.stack(local_map)
    local_ic = np.stack(local_ic)
    local_homo = torch.stack(local_homo)



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



class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, data_split, device='cpu', scale=1
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = os.path.join(data_dir, data_split)
        self.device = device
        self.scale = scale

        with open(os.path.join(data_dir, data_split + '.pkl'), 'rb') as handle:
            all_data = pickle.load(handle)


        self.obs_frame_num = all_data['obs_frame_num']
        self.fut_frame_num = all_data['fut_frame_num']

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(all_data['obs_traj']).float().to(self.device)
        self.fut_traj = torch.from_numpy(all_data['fut_traj']).float().to(self.device)

        self.seq_start_end = all_data['seq_start_end']
        self.map_file_name = all_data['map_file_name']
        self.inv_h_t = all_data['inv_h_t']

        self.local_map = all_data['local_map']
        self.local_homo = all_data['local_homo']
        self.local_ic = all_data['local_ic']

        self.num_seq = len(self.seq_start_end)
        print(self.seq_start_end[-1])


    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        out = [
            self.obs_traj[index, :], self.fut_traj[index, :],
            self.map_file_name[index], self.inv_h_t[index],
            self.local_map[index],
            self.local_ic[index],
            torch.from_numpy(self.local_homo[index]).float().to(self.device), self.scale
        ]
        # start, end = self.seq_start_end[index]
        # out = [
        #     self.obs_traj[start:end, :], self.fut_traj[start:end, :],
        #     np.array([self.map_file_name[index]] * (end - start)), np.array([self.inv_h_t[index]] * (end - start)),
        #     self.local_map[start:end],
        #     self.local_ic[start:end],
        #     torch.from_numpy(self.local_homo[start:end]).float().to(self.device), self.scale
        # ]
        return out


