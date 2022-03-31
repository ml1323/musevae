import copy
import random
import numpy as np
import torch
from data.nuscenes.nuscenes_pred_split import get_nuscenes_pred_split
from data.nuscenes.nuscenes_preprocessor import preprocess
from util import derivative_of


class data_generator(object):

    def __init__(self, parser, log, split='train', phase='training', batch_size=8, device='cpu', scale=1, shuffle=False):
        self.past_frames = parser.past_frames
        self.min_past_frames = parser.min_past_frames

        self.frame_skip = parser.get('frame_skip', 1)
        self.phase = phase
        self.split = split
        self.device = device
        self.scale = scale
        self.batch_size = batch_size
        self.max_train_agent = parser.get('max_train_agent', 32)

        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        data_root = parser.data_root_nuscenes_pred
        seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
        self.init_frame = 0
        process_func = preprocess
        self.data_root = data_root

        print("\n-------------------------- loading %s data --------------------------" % split)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else: assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        for seq_name in self.sequence_to_load:
            # print("loading sequence {} ...".format(seq_name))
            # ns_scene = nusc.get('scene', nusc.field2token('scene', 'name', ns_scene_name)[0])

            preprocessor = process_func(data_root, seq_name, parser, log, self.split, self.phase)

            num_seq_samples = preprocessor.num_fr - (parser.min_past_frames - 1) * self.frame_skip - parser.min_future_frames * self.frame_skip + 1
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
            # print(num_seq_samples)
        print("{} sequence files are loaded ...".format(len(self.sequence_to_load)))
        self.sample_list = list(range(self.num_total_samples))
        self.idx_list = []
        for i in self.sample_list[::self.batch_size]:
            if i==self.num_total_samples //self.batch_size * self.batch_size:
                self.idx_list.append((i, self.sample_list[-1]+1))
            else:
                self.idx_list.append((i, i+self.batch_size))

        if shuffle:
            random.shuffle(self.sample_list)

        self.index = 0
        self.local_ic = [[]] * len(self.sample_list)
        self.local_homo = [[]] * len(self.sample_list)
        print(f'total num samples: {self.num_total_samples}')
        print("------------------------------ done --------------------------------\n")

        
    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self, force=False):
        if (self.index+1 >= len(self.idx_list)) or force:
            self.index = 0      # reset
            return True
        else:
            return False

    def next_sample(self):
        obs_traj = []
        fut_traj = []
        rng_idx = self.idx_list[self.index]
        self.index += 1
        local_maps = []
        local_ics = []
        local_homos = []
        scene_maps = []
        seq_names = []
        for sample_index in self.sample_list[rng_idx[0]: rng_idx[1]]:

            seq_index, frame = self.get_seq_and_frame(sample_index)
            seq = self.sequence[seq_index]
            # get valid seq
            data = seq(frame)
            if data is None:
                continue

            in_data = data
            obs_traj.append(torch.stack(in_data['pre_motion_3D']))
            fut_traj.append(torch.stack(in_data['fut_motion_3D']))

            all_traj = torch.cat([obs_traj[-1], fut_traj[-1]], dim=1)
            # get local map
            scene_map = data['scene_map']
            scene_points = all_traj * data['traj_scale']
            radius = []
            for i in range(len(all_traj)):
                map_traj = scene_map.to_map_points(scene_points[i])

                r = np.clip(np.sqrt(((map_traj[1:] - map_traj[:-1]) ** 2).sum(1)).mean() * 20, a_min=128, a_max=None)
                radius.append(np.round(r).astype(int))
                seq_names.append(seq.seq_name)
            comput_local_homo = (len(self.local_ic[sample_index]) == 0)
            local_map, local_ic, local_homo = scene_map.get_cropped_maps(scene_points, radius, compute_local_homo=comput_local_homo)
            if comput_local_homo:
                self.local_ic[sample_index] = np.stack(local_ic)
                self.local_homo[sample_index] = np.stack(local_homo)
            local_maps.extend(local_map)
            local_ics.append(self.local_ic[sample_index])
            local_homos.append(self.local_homo[sample_index])
            scene_maps.append(scene_map)

        if len(obs_traj) == 0:
            return None

        local_ics = np.concatenate(local_ics)
        local_homos = np.concatenate(local_homos)

        _len = [len(seq) for seq in obs_traj]
        cum_start_idx = [0] + np.cumsum(_len).tolist()
        seq_start_end = [[start, end]
                         for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        obs_traj = torch.cat(obs_traj)
        fut_traj = torch.cat(fut_traj)
        all_traj = torch.cat([obs_traj, fut_traj], dim=1)

        # 6 states
        all_stat = []
        dt = 0.5
        for one_seq in all_traj.detach().cpu().numpy():
            x = one_seq[:, 0].astype(float)
            y = one_seq[:, 1].astype(float)
            vx = derivative_of(x, dt)
            vy = derivative_of(y, dt)
            ax = derivative_of(vx, dt)
            ay = derivative_of(vy, dt)
            all_stat.append(np.stack([x, y, vx, vy, ax, ay]))
        all_stat = torch.tensor(np.stack(all_stat)).float().to(self.device).permute(2,0,1)

        # get vel and acc
        obs_traj = all_stat[:self.past_frames]
        fut_traj = all_stat[self.past_frames:]


        obs_traj_st = obs_traj.clone()
        # pos is stdized by mean = last obs step
        obs_traj_st[:, :, :2] = (obs_traj_st[:, :, :2] - obs_traj_st[-1, :, :2]) / self.scale
        obs_traj_st[:, :, 2:] /= self.scale
        out = [
            obs_traj, fut_traj, obs_traj_st, fut_traj[:, :, 2:4] / self.scale, seq_start_end,
            scene_maps, None, local_maps, local_ics, torch.tensor(local_homos).float().to(self.device)
        ]

        return out
