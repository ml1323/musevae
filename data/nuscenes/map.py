"""
Code borrowed from Trajectron++: https://github.com/StanfordASL/Trajectron-plus-plus/blob/ef0165a93ee5ba8cdc14f9b999b3e00070cd8588/trajectron/environment/map.py
"""

import torch
import numpy as np
import cv2
import os
from .homography_warper import get_rotation_matrix2d, warp_affine_crop


class Map(object):
    def __init__(self, data, homography, description=None):
        self.data = data
        self.homography = homography
        self.description = description

    def as_image(self):
        raise NotImplementedError

    def get_cropped_maps(self, world_pts, patch_size, rotation=None, device='cpu'):
        raise NotImplementedError

    def to_map_points(self, scene_pts):
        raise NotImplementedError


class GeometricMap(Map):
    """
    A Geometric Map is a int tensor of shape [layers, x, y]. The homography must transform a point in scene
    coordinates to the respective point in map coordinates.

    :param data: Numpy array of shape [layers, x, y]
    :param homography: Numpy array of shape [3, 3]
    """
    def __init__(self, data, homography, origin=None, description=None):
        #assert isinstance(data.dtype, np.floating), "Geometric Maps must be float values."
        super(GeometricMap, self).__init__(data, homography, description=description)

        if origin is None:
            self.origin = np.zeros(2)
        else:
            self.origin = origin
        self._last_padded_map = None
        self._torch_map = None

    def torch_map(self, device):
        if self._torch_map is not None:
            return self._torch_map
        self._torch_map = torch.tensor(self.data, dtype=torch.uint8, device=device)
        return self._torch_map

    def as_image(self):
        # We have to transpose x and y to rows and columns. Assumes origin is lower left for image
        # Also we move the channels to the last dimension
        return (np.transpose(self.data, (2, 1, 0))).astype(np.uint)

    def get_padded_map(self, padding_x, padding_y):
        data = 1 - self.data / 255
        data[0][np.where(data[1] == 0)] = 0.3
        data[0][np.where(data[2] == 0)] = 0.6
        self._last_padded_map = np.full((self.data.shape[1] + 2 * padding_x,
                                         self.data.shape[2] + 2 * padding_y),
                                        1, dtype=np.float32)
        self._last_padded_map[padding_x:-padding_x, padding_y:-padding_y] = data[0]
        return self._last_padded_map

    @staticmethod
    def batch_rotate(map_batched, centers, angles, out_height, out_width):
        """
        As the input is a map and the warp_affine works on an image coordinate system we would have to
        flip the y axis updown, negate the angles, and flip it back after transformation.
        This, however, is the same as not flipping at and not negating the radian.

        :param map_batched:
        :param centers:
        :param angles:
        :param out_height:
        :param out_width:
        :return:
        """
        M = get_rotation_matrix2d(centers, angles, torch.ones_like(angles))
        rotated_map_batched = warp_affine_crop(map_batched, centers, M,
                                               dsize=(out_height, out_width), padding_mode='zeros')

        return rotated_map_batched

    @classmethod
    def get_cropped_maps_from_scene_map_batch(cls, maps, scene_pts, radius, compute_local_homo=False):
        """
        Returns rotated patches of each map around the transformed scene points.
        ___________________
        |       |          |
        |       |ps[3]     |
        |       |          |
        |       |          |
        |      o|__________|
        |       |    ps[2] |
        |       |          |
        |_______|__________|
        ps = patch_size

        :param maps: List of GeometricMap objects [bs]
        :param scene_pts: Scene points: [bs, 2]
        :param patch_size: Extracted Patch size after rotation: [-x, -y, +x, +y]
        :param rotation: Rotations in degrees: [bs]
        :param device: Device on which the rotated tensors should be returned.
        :return: Rotated and cropped tensor patches.
        """
        scene_pts = scene_pts.detach().cpu().numpy()
        padded_map = []
        for i in range(len(maps)):
            padded_map.append(maps[i].get_padded_map(radius[i], radius[i]))

        local_maps = []
        local_homos = []
        local_ics = []
        for agent_idx in range(len(padded_map)):
            r = radius[agent_idx]
            expanded_obs_img= padded_map[agent_idx]
            all_traj = scene_pts[agent_idx, :4]
            all_pixel = maps[agent_idx].to_map_points(all_traj)
            all_pixel = r + np.round(all_pixel).astype(int)
            local_map = expanded_obs_img[all_pixel[-1, 0] - r: all_pixel[-1, 0] + r,
                        all_pixel[-1, 1] - r: all_pixel[-1, 1] + r]
            local_maps.append(local_map)

            if compute_local_homo:
                fake_pt = [all_traj[-1]]
                per_pixel_dist = r // 10
                for i in range(per_pixel_dist, r // 3 - per_pixel_dist, per_pixel_dist):
                    fake_pt.append(all_traj[-1] + [i, i] + np.random.rand(2) * (per_pixel_dist // 4))
                    fake_pt.append(all_traj[-1] + [-i, -i] + np.random.rand(2) * (per_pixel_dist // 4))
                    fake_pt.append(all_traj[-1] + [i, -i] + np.random.rand(2) * (per_pixel_dist // 4))
                    fake_pt.append(all_traj[-1] + [-i, i] + np.random.rand(2) * (per_pixel_dist // 4))
                fake_pt = np.array(fake_pt)

                fake_pixel = maps[agent_idx].to_map_points(fake_pt)
                fake_pixel = r + np.round(fake_pixel).astype(int)


                temp_map_val = []
                for i in range(len(fake_pixel)):
                    temp_map_val.append(expanded_obs_img[fake_pixel[i, 0], fake_pixel[i, 1]])
                    expanded_obs_img[fake_pixel[i, 0], fake_pixel[i, 1]] = i + 10

                fake_local_pixel = []
                for i in range(len(fake_pixel)):
                    fake_local_pixel.append([np.where(local_map == i + 10)[0][0], np.where(local_map == i + 10)[1][0]])
                    expanded_obs_img[fake_pixel[i, 0], fake_pixel[i, 1]] = temp_map_val[i]

                h, _ = cv2.findHomography(np.array([fake_local_pixel]), np.array(fake_pt))

                all_traj = scene_pts[agent_idx]
                all_pixel_local = np.matmul(np.concatenate([all_traj, np.ones((len(all_traj), 1))], axis=1),
                                            np.linalg.pinv(np.transpose(h)))
                all_pixel_local /= np.expand_dims(all_pixel_local[:, 2], 1)
                all_pixel_local = np.round(all_pixel_local).astype(int)[:, :2]

                local_ics.append(all_pixel_local)
                local_homos.append(h)

        return local_maps, local_ics, local_homos



    def get_cropped_maps(self, scene_pts, patch_size, compute_local_homo=False):
        """
        Returns rotated patches of the map around the transformed scene points.
        ___________________
        |       |          |
        |       |ps[3]     |
        |       |          |
        |       |          |
        |      o|__________|
        |       |    ps[2] |
        |       |          |
        |_______|__________|
        ps = patch_size

        :param scene_pts: Scene points: [bs, 2]
        :param patch_size: Extracted Patch size after rotation: [-lat, -long, +lat, +long]
        :param rotation: Rotations in degrees: [bs]
        :param device: Device on which the rotated tensors should be returned.
        :return: Rotated and cropped tensor patches.
        """
        return self.get_cropped_maps_from_scene_map_batch([self]*scene_pts.shape[0], scene_pts,
                                                          patch_size, compute_local_homo=compute_local_homo)

    def to_map_points(self, scene_pts):
        org_shape = None
        if len(scene_pts.shape) != 2:
            org_shape = scene_pts.shape
            scene_pts = scene_pts.reshape((-1, 2))
        scene_pts = scene_pts - self.origin[None, :]
        N, dims = scene_pts.shape
        points_with_one = np.ones((dims + 1, N))
        points_with_one[:dims] = scene_pts.T
        map_points = (self.homography @ points_with_one).T[..., :dims]
        if org_shape is not None:
            map_points = map_points.reshape(org_shape)
        return map_points


    def visualize_data(self, data):
        pre_motion = np.stack(data['pre_motion_3D']) * data['traj_scale']
        fut_motion = np.stack(data['fut_motion_3D']) * data['traj_scale']
        heading = data['heading']
        img = np.transpose(self.data, (1, 2, 0))
        for i in range(pre_motion.shape[0]):
            cur_pos = pre_motion[i, -1]
            # draw agent
            cur_pos = np.round(self.to_map_points(cur_pos)).astype(int)
            img = cv2.circle(img, (cur_pos[1], cur_pos[0]), 3, (0, 255, 0), -1)
            prev_pos = cur_pos
            # draw fut traj
            for t in range(fut_motion.shape[0]):
                pos = fut_motion[i, t]
                pos = np.round(self.to_map_points(pos)).astype(int)
                img = cv2.line(img, (prev_pos[1], prev_pos[0]), (pos[1], pos[0]), (0, 255, 0), 2) 

            # draw heading
            theta = heading[i]
            v= np.array([5.0, 0.0])
            v_new = v.copy()
            v_new[0] = v[0] * np.cos(theta) - v[1] * np.sin(theta)
            v_new[1] = v[0] * np.sin(theta) + v[1] * np.cos(theta)
            vend = pre_motion[i, -1] + v_new
            vend = np.round(self.to_map_points(vend)).astype(int)
            img = cv2.line(img, (cur_pos[1], cur_pos[0]), (vend[1], vend[0]), (0, 255, 255), 2) 

        fname = f'out/agent_maps/{data["seq"]}_{data["frame"]}_vis.png'
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        cv2.imwrite(fname, img)

