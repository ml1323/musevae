import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal


class EncoderX(nn.Module):
    def __init__(
        self, zS_dim, enc_h_dim=64, mlp_dim=32, map_feat_dim=32, map_mlp_dim=32,
            num_layers=1, dropout_mlp=0.3, dropout_rnn=0.25, device='cpu', map_h_dim=10240,
    ):
        super(EncoderX, self).__init__()

        self.zS_dim=zS_dim
        self.enc_h_dim = enc_h_dim
        self.num_layers = num_layers
        self.dropout_rnn=dropout_rnn
        self.dropout_mlp=dropout_mlp

        n_state=6

        self.rnn_encoder = nn.LSTM(
            input_size=n_state, hidden_size=enc_h_dim
        )


        self.fc1 = nn.Linear(enc_h_dim, mlp_dim)
        self.fc_hidden = nn.Linear(mlp_dim + map_feat_dim, mlp_dim)
        self.fc_latent = nn.Linear(mlp_dim, zS_dim*2)

        self.map_h_dim = map_h_dim

        self.map_fc1 = nn.Linear(self.map_h_dim + 9, map_mlp_dim)
        self.map_fc2 = nn.Linear(map_mlp_dim, map_feat_dim)


    def forward(self, obs_traj, seq_start_end, local_map_feat, local_homo, train=False):
        _, (final_encoder_h, _) = self.rnn_encoder(obs_traj)
        final_encoder_h = F.dropout(final_encoder_h,
                            p=self.dropout_rnn,
                            training=train)
        hx = self.fc1(final_encoder_h.view(-1, self.enc_h_dim))
        hx = F.dropout(F.relu(hx),
                      p=self.dropout_mlp,
                      training=train)
        stats = self.fc_latent(hx)

        local_map_feat = local_map_feat.view(-1, self.map_h_dim)
        local_homo = local_homo.view(-1, 9)
        map_feat = self.map_fc1(torch.cat((local_map_feat, local_homo), dim=-1))
        map_feat = F.dropout(F.relu(map_feat),
                      p=self.dropout_mlp,
                      training=train)
        map_feat = self.map_fc2(map_feat)

        hx = self.fc_hidden(torch.cat((hx, map_feat), dim=-1))

        return hx, stats[:,:self.zS_dim], stats[:,self.zS_dim:]


class EncoderY(nn.Module):
    def __init__(
            self, zS_dim, enc_h_dim=64, mlp_dim=32,
            num_layers=1, dropout_mlp=0.3, dropout_rnn=0.25,
            device='cpu'
    ):
        super(EncoderY, self).__init__()

        self.zS_dim = zS_dim
        self.enc_h_dim = enc_h_dim
        self.num_layers = num_layers
        self.device = device
        n_state = 6
        n_pred_state = 2
        self.dropout_rnn = dropout_rnn
        self.dropout_mlp = dropout_mlp

        self.initial_h_model = nn.Linear(n_state, enc_h_dim)
        self.initial_c_model = nn.Linear(n_state, enc_h_dim)
        self.rnn_encoder = nn.LSTM(
            input_size=n_pred_state, hidden_size=enc_h_dim, num_layers=num_layers, bidirectional=True
        )

        self.fc1 = nn.Linear(4*enc_h_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, zS_dim*2)

    def forward(self, last_obs_traj, fut_vel, seq_start_end, obs_enc_feat, train=False):

        initial_h = self.initial_h_model(last_obs_traj)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = self.initial_c_model(last_obs_traj)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)
        state_tuple = (initial_h, initial_c)

        _, state = self.rnn_encoder(fut_vel, state_tuple)

        final_encoder_h = torch.cat(state, dim=0).permute(1, 0, 2)
        final_encoder_h = F.dropout(final_encoder_h,
                                    p=self.dropout_rnn,
                                    training=train)


        stats = self.fc1(final_encoder_h.reshape(-1, 4 * self.enc_h_dim))
        stats = F.dropout(F.relu(stats),
                      p=self.dropout_mlp,
                      training=train)
        stats = self.fc2(stats)

        return stats[:,:self.zS_dim], stats[:,self.zS_dim:]


class Decoder(nn.Module):
    def __init__(
        self, seq_len, dec_h_dim=128, mlp_dim=1024, num_layers=1,
        dropout_rnn=0.25, enc_h_dim=32, z_dim=32,
        device='cpu', scale=1, dt=0.4
    ):
        super(Decoder, self).__init__()
        n_state=6
        n_pred_state=2
        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.dec_h_dim = dec_h_dim
        self.enc_h_dim = enc_h_dim
        self.device=device
        self.num_layers = num_layers
        self.dropout_rnn = dropout_rnn
        self.scale = scale
        self.dt = dt

        self.dec_hidden = nn.Linear(mlp_dim + z_dim, dec_h_dim)
        self.to_vel = nn.Linear(n_state, n_pred_state)

        self.rnn_decoder = nn.GRUCell(
            input_size=mlp_dim + z_dim + 2*n_pred_state, hidden_size=dec_h_dim
        )

        self.fc_mu = nn.Linear(dec_h_dim, n_pred_state)
        self.fc_std = nn.Linear(dec_h_dim, n_pred_state)

        self.sg_rnn_enc = nn.LSTM(
            input_size=n_state, hidden_size=enc_h_dim, num_layers=num_layers, bidirectional=True)
        self.sg_fc = nn.Linear(4*enc_h_dim, n_pred_state)


    def forward(self, last_obs_st, last_obs_pos, enc_h_feat, z, sg, sg_update_idx, fut_vel_st=None):

        zx = torch.cat([enc_h_feat, z], dim=1)
        decoder_h=self.dec_hidden(zx)
        pred_vel = self.to_vel(last_obs_st)

        dt = self.dt * (self.seq_len/len(sg_update_idx))
        last_ob_sg = torch.cat([last_obs_pos.unsqueeze(1), sg], dim=1).detach().cpu().numpy()
        last_ob_sg = (last_ob_sg - last_ob_sg[:,:1])/self.scale

        sg_state = []
        for pos in last_ob_sg:
            vx = np.gradient(pos[:,0], dt)
            vy = np.gradient(pos[:,1], dt)
            ax = np.gradient(vx, dt)
            ay = np.gradient(vy, dt)
            sg_state.append(np.array([pos[:,0], pos[:,1], vx, vy, ax, ay]))
        sg_state = torch.tensor(np.stack(sg_state)).permute((2,0,1)).float().to(z.device)

        ### sg encoding
        _, sg_h = self.sg_rnn_enc(sg_state)
        sg_h = torch.cat(sg_h, dim=0).permute(1, 0, 2)
        if fut_vel_st is not None:
            train=True
        else:
            train=False
        sg_h = F.dropout(sg_h,
                        p=self.dropout_rnn,
                        training=train)
        sg_feat = self.sg_fc(sg_h.reshape(-1, 4 * self.enc_h_dim))


        ### traj decoding
        mus = []
        stds = []
        j=0
        for i in range(self.seq_len):
            decoder_h= self.rnn_decoder(torch.cat([zx, pred_vel, sg_feat], dim=1), decoder_h)
            mu= self.fc_mu(decoder_h)
            logVar = self.fc_std(decoder_h)
            std = torch.sqrt(torch.exp(logVar))
            mus.append(mu)
            stds.append(std)

            if train:
                pred_vel = fut_vel_st[i]
            else:
                if(i == sg_update_idx[j]):
                    pred_vel = sg_state[j+1,:,2:4]
                    j += 1
                else:
                    pred_vel = Normal(mu, std).rsample()

        mus = torch.stack(mus, dim=0)
        stds = torch.stack(stds, dim=0)
        return Normal(mus, stds)


