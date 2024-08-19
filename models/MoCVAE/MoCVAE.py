import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal
from argparse import Namespace
from models.ConvAutoEncoder.auto_encoder import ConvAutoEncoder
from models.MoCVAE.VAE.encoder import Encoder
from models.MoCVAE.VAE.decoder import Decoder

from models.MoCVAE.GAT.gat_model import GAT
from models.MoCVAE.GCN.gcn_model import GCN
import numpy as np
import random
from utils.adjacency_matrix import (
    compute_vae_adjs_distsim,
    adjs_fully_connected_pred,
    adjs_distance_sim_pred,
    adjs_knn_sim_pred,
)
from utils.eval import mean_square_error, average_displacement_error, final_displacement_error, sample_multinomial


class HittingNet(nn.Module):
    """
    Hitting Net aims to predict who will hit the ball next by one-hot encoding.

    It takes the hidden state h_t and the previous hitting player feature P_{t-1} as input and outputs P_t.
    """

    def __init__(self, p_dim: int, rnn_dim: int):
        super(HittingNet, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(p_dim+rnn_dim , p_dim*2),
            nn.ReLU(),
            nn.Linear(p_dim*2, p_dim),
            nn.Softmax(dim=-1),
        )
        self.hit_linear = nn.Sequential(
            nn.Linear(p_dim, p_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(p_dim * 2, p_dim),
        )
        self.hit_comb = nn.Sequential(nn.Linear(p_dim * 2, p_dim), nn.Softmax(dim=-1))

    def forward(self, p_t_1: Tensor, h) -> Tensor:
        out = self.nn(torch.cat([p_t_1,h], 1))
        # hit_transform = self.hit_linear(out)
        # return self.hit_comb(torch.cat((out, hit_transform), dim=-1))
        return out


class ShotTypeNet(nn.Module):
    """
    Shot Type Net aims to predict the type of the shot by one-hot encoding.

    It takes the hidden state h_t, the previous shot type feature S_{t-1} and the previous hitting player feature P_{t-1} as input and outputs S_t.
    """

    def __init__(self, p_dim: int, shot_type_dim: int, rnn_dim: int):
        super(ShotTypeNet, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(p_dim + shot_type_dim +rnn_dim, shot_type_dim),
            nn.ReLU(),
            nn.Linear(shot_type_dim, shot_type_dim),
            nn.Softmax(dim=-1),
        )
        self.shot_type_linear = nn.Sequential(
            nn.Linear(shot_type_dim, shot_type_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(shot_type_dim * 2, shot_type_dim),
        )
        self.shot_type_comb = nn.Sequential(nn.Linear(shot_type_dim * 2, shot_type_dim), nn.Softmax(dim=-1))

    def forward(self, s_t_1: Tensor, p_t_1: Tensor, h: Tensor) -> Tensor:
        out = self.nn(torch.cat([s_t_1, p_t_1, h], 1))
        # shot_transfrom = self.shot_type_linear(out)
        # return self.shot_type_comb(torch.cat((out, shot_transfrom), dim=-1))
        return out

class MoCVAE(nn.Module):
    def __init__(self, args: Namespace, n_max_player: int):
        super(MoCVAE, self).__init__()

        self.n_max_player: int = n_max_player
        self.n_layers: int = args.n_layers
        self.x_dim: int = args.x_dim
        self.h_dim: int = args.h_dim
        self.z_dim: int = args.z_dim
        self.p_dim: int = args.p_dim
        self.d_dim: int = n_max_player * 2
        self.s_dim: int = args.s_dim
        self.rnn_dim: int = args.rnn_dim
        self.batch_size: int = args.batch_size
        self.obs_len: int = args.obs_len
        self.pred_len: int = args.pred_len

        self.graph_model: str = args.graph_model
        self.graph_hid: int = args.graph_hid
        self.adjacency_type: str = args.adjacency_type
        self.top_k_neigh: int = args.top_k_neigh
        self.sigma = args.sigma
        self.alpha: float = args.alpha
        self.n_heads: int = args.n_heads
        self.device: str = args.device

        # Checking
        if self.adjacency_type == 2:
            assert self.top_k_neigh is not None, "Using KNN-similarity but top_k_neigh is not specified"

        # Embeddings
        self.position_emb = ConvAutoEncoder.from_checkpoint(args.position_emb_path, eval=True)
        self.prior_emb = nn.Conv1d(in_channels=self.obs_len, out_channels=16, kernel_size=3, padding=1)
        self.encode_emb = nn.Conv1d(in_channels=self.obs_len + self.pred_len, out_channels=16, kernel_size=3, padding=1)
        self.history = nn.Linear(self.rnn_dim * 2, self.rnn_dim)

        # Feature Extractors
        self.phi_x_dec = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(),
        )
        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(),
        )

        # Shot Type Net
        self.shot_type_net = ShotTypeNet(p_dim=self.p_dim, shot_type_dim=self.s_dim, rnn_dim=self.rnn_dim)

        # Hit player Net
        self.hit_player_net = HittingNet(p_dim=self.p_dim, rnn_dim=self.rnn_dim)

        # RNN
        self.rnn = nn.GRU(self.h_dim * 2, self.rnn_dim, self.n_layers)

        # Encoders
        self.gat_enc = GAT(
            in_dim=self.x_dim,
            hidden_dim=self.graph_hid,
            out_dim=self.x_dim,
            alpha=self.alpha,
            n_heads=self.n_heads,
        )
        # self.enc = Encoder(
        #     in_dim=(self.p_dim + self.h_dim + self.x_dim * 16 + self.s_dim + self.rnn_dim),
        #     out_dim=self.z_dim,
        #     hidden_dim=self.h_dim,
        # )
        self.prior = Encoder(
            in_dim=(self.p_dim + self.x_dim * 16 + self.s_dim + self.rnn_dim + self.h_dim),
            out_dim=self.z_dim,
            hidden_dim=self.h_dim,
        )

        # Decoders
        self.gat_dec = GAT(
            in_dim=self.z_dim,
            hidden_dim=self.graph_hid,
            out_dim=4,
            alpha=self.alpha,
            n_heads=self.n_heads,
        )
        self.dec = Decoder(
            in_dim= self.rnn_dim,
            out_dim=self.x_dim,
            hidden_dim=self.h_dim,
        )

        # Generator
        if self.graph_model == "gcn":
            self.graph_hiddens = GCN(in_dim=self.rnn_dim, out_dim=self.rnn_dim, hidden_dim=self.graph_hid)
        elif self.graph_model == "gat":
            self.graph_hiddens = GAT(in_dim=self.rnn_dim, out_dim=self.rnn_dim, hidden_dim=self.graph_hid,alpha=self.alpha, n_heads=self.n_heads)  # fmt: skip
        else:
            raise Exception(f"Invalid graph model: {self.graph_model}")

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    def _nll_gauss(self, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        x1 = torch.sum(((x - mu).pow(2)) / torch.exp(logvar), dim=1)
        x2 = x.size(1) * np.log(2 * np.pi)
        x3 = torch.sum(logvar, dim=1)
        nll = torch.mean(0.5 * (x1 + x2 + x3))
        return nll

    def _kld(self, mean_enc, logvar_enc, mean_prior, logvar_prior):
        x1 = torch.sum((logvar_prior - logvar_enc), dim=1)
        x2 = torch.sum(torch.exp(logvar_enc - logvar_prior), dim=1)
        x3 = torch.sum((mean_enc - mean_prior).pow(2) / (torch.exp(logvar_prior)), dim=1)
        kld_element = x1 - mean_enc.size(1) + x2 + x3
        return torch.mean(0.5 * kld_element)

    def forward(
        self,
        traj: Tensor,
        traj_rel: Tensor,
        shot_type_ohe: Tensor,
        hit_player: Tensor,
        seq_start_end: Tensor,
        adj_out: Tensor,
        mask: bool,
        tf_threshold: float,
        obs_len: int,
    ):
        timesteps, batch, features = traj.shape
        
        x_in_conv = traj
        pos_emb = traj.reshape(timesteps, -1, 4, features).permute(1, 0, 2, 3).to(self.device)
        
        traj_with_emb = []
        for i in range(pos_emb.shape[2]):
            out, _ = self.position_emb(pos_emb[:, :, i, :])
            traj_with_emb.append(out.squeeze(1))
        traj_with_emb = torch.stack(traj_with_emb)
        x_emb = traj_rel.reshape(timesteps, -1, 4, 2).permute(1, 0, 2, 3).to(self.device)
        add_emb = []
        for i, player_emb in enumerate(traj_with_emb):
            new_coords = x_emb[:, :, i, :] + player_emb
            add_emb.append(new_coords)
        x_emb = torch.stack(add_emb).permute(1, 0, 2, 3).reshape(timesteps, -1, 2)
        if mask:
            x_emb = x_emb[:obs_len, :, :]
            traj = traj[:obs_len, :, :]
            traj_rel = traj_rel[:obs_len, :, :]
            timesteps, batch, features = traj.shape

        d = torch.zeros(timesteps, batch, features * self.n_max_player).to(self.device)
        h = torch.zeros(self.n_layers, batch, self.rnn_dim).to(self.device)

        # an agent has to know all the xy abs positions of all the other agents in its sequence (for every timestep)
        for idx, (start, end) in enumerate(seq_start_end):
            n_agents = (end - start).item()
            d[:, start:end, : n_agents * 2] = (
                traj[:, start:end, :].reshape(timesteps, -1).unsqueeze(1).repeat(1, n_agents, 1)
            )
        # KLD = torch.zeros((timesteps, batch, self.z_dim)).to(self.device)
        KLD = torch.zeros(1).to(self.device)
        NLL = torch.zeros(1).to(self.device)
        mse_value = torch.zeros(1).to(self.device)
        ade_value = torch.zeros(1).to(self.device)
        fde_value = torch.zeros(1).to(self.device)
        shot = []
        hit = []
        x_pred_real = traj[0]
        x_pred_next = traj_rel[1]

        x_in_conv = x_in_conv.reshape(x_in_conv.shape[0], batch // 4, 4, features).permute(1, 2, 3, 0).contiguous()
        x_in_conv = x_in_conv.view(batch // 4, 4 * features, -1)
        pred = []
        for timestep in range(1, timesteps):
            x_real = traj[timestep - 1]
            x_t = traj_rel[timestep]
            x_n = traj_rel[timestep]

            t = random.random()
            if t > tf_threshold:
                x_emb_t = x_pred_next

            x_emb_t = x_emb[timestep]
            shot_type_t = shot_type_ohe[timestep]  # ground truth goal
            h_t = hit_player[timestep]
            if (shot_type_t == 0).all():
                return KLD, NLL, cross_entropy, h
            vae_adj_encod_out = compute_vae_adjs_distsim(
                1.2,
                seq_start_end,
                x_t.detach().cpu(),
            ).to(self.device)

            

            gat_t = self.gat_enc(x_emb_t.clone(), vae_adj_encod_out)
           
            phi_x_t = self.phi_x(gat_t)  # with embedding
            if timestep < self.obs_len + 1:
                phi_pri_x_t = phi_x_t
            else:
                phi_pri_x_t = torch.zeros_like(phi_x_t).to(self.device)
            if mask:
                past_encoder_x = x_emb.permute(1, 0, 2).contiguous()

                padding_num = self.obs_len + self.pred_len - past_encoder_x.shape[1]
                padding_ = torch.zeros(past_encoder_x.shape[0], padding_num, past_encoder_x.shape[2]).to(self.device)
                past_encoder_x = torch.cat([past_encoder_x, padding_], dim=1)
                encoder_emb_x = self.encode_emb(past_encoder_x).reshape(past_encoder_x.shape[0], -1)
            else:
                past_encoder_x = x_emb.permute(1, 0, 2).contiguous()
                encoder_emb_x = self.encode_emb(past_encoder_x).reshape(past_encoder_x.shape[0], -1)

            # cross_entropy += F.binary_cross_entropy(shot_combined, shot_type_t, reduction="sum")
            # cross_entropy_hit += F.binary_cross_entropy(hit_combined, h_t, reduction="sum")

             # prior
            past_prior_x = traj[: self.obs_len].permute(1, 0, 2).contiguous()
            padding_num = self.obs_len - past_prior_x.shape[1]
            padding_ = torch.zeros(past_prior_x.shape[0], padding_num, past_prior_x.shape[2]).to(self.device)
            past_prior_x = torch.cat([past_prior_x, padding_], dim=1)

            prior_emb_x = self.prior_emb(past_prior_x)
            prior_emb_x = prior_emb_x.reshape(traj.shape[1], -1)

           
            hit_combined = self.hit_player_net(hit_player[timestep - 1],h[-1])
            shot_combined = self.shot_type_net(shot_type_ohe[timestep - 1], hit_combined, h[-1])
            shot.append(shot_combined)
            hit.append(hit_combined)

            prior_mean_t, prior_logvar_t = self.prior(
                torch.cat([prior_emb_x, phi_pri_x_t, shot_combined, hit_combined, h[-1]], 1)
            )  # prior_emb_x

            enc_mean_t, enc_logvar_t = self.enc(
                torch.cat([encoder_emb_x, phi_x_t, shot_combined, hit_combined, h[-1]], 1)
            )  # encoder_emb_x

           
            # sampling from latent
            z_t = self._reparameterize(enc_mean_t, enc_logvar_t)
            phi_z_t = self.phi_z(z_t)
            dec_mean_t, dec_logvar_t = self.dec(phi_z_t)

            # agent vrnn recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

         
            KLD += self._kld(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
            NLL += self._nll_gauss(dec_mean_t, dec_logvar_t, x_t)
            x_pred_real = dec_mean_t + x_real
            pred.append(x_pred_real)

            h_graph = self.graph_hiddens(h[-1].clone(), adj_out[timestep])  # graph refinement

            h[-1] = self.history(torch.cat((h_graph, h[-1]), dim=-1)).unsqueeze(0)  # combination
        pred = torch.stack(pred)

        shot = torch.stack(shot)
        hit = torch.stack(hit)

        cross_entropy = F.binary_cross_entropy(shot, shot_type_ohe[1:], reduction="sum") / (shot.shape[0] * shot.shape[1])
        cross_entropy_hit = F.binary_cross_entropy(hit, hit_player[1:], reduction="sum") / (hit.shape[0] * hit.shape[1])

        mse_value = mean_square_error(pred, traj[1:], pred_len=pred.shape[0])
        ade_value = average_displacement_error(pred, traj[1:], pred_len=pred.shape[0])
        fde_value = final_displacement_error(pred, traj[1:], pred_len=pred.shape[0])
        return (
            # KLD.sum(2).mean(),
            KLD,
            NLL,
            cross_entropy,
            cross_entropy_hit,
            h,
            mse_value,
            ade_value,
            fde_value,
        )

    def sample(
        self,
        samples_seq_len: int,
        h: Tensor,
        x_abs_start: Tensor,
        shot_type_start: Tensor,
        hit_start: Tensor,
        seq_start_end: Tensor,
    ):
        # _, batch_size, _ = h.shape
        batch_size, _ = x_abs_start.shape
        s_t = shot_type_start
        x_t_abs = x_abs_start
        hit_t = hit_start
        samples = torch.zeros(samples_seq_len, batch_size, self.x_dim).cuda()
        displacements = torch.zeros(samples_seq_len, batch_size, self.n_max_player * 2).cuda()

        x_pred_next = x_t_abs
        pred_traj = torch.zeros(samples_seq_len + 1, batch_size, self.x_dim).cuda()
        pred_traj[0] = x_abs_start
        vae_adj_encod_out = compute_vae_adjs_distsim(
            1.2,
            seq_start_end,
            x_t_abs.detach().cpu(),
        ).cuda()
        hit_pred = []
        shot_pred = []
        with torch.no_grad():
            for timestep in range(samples_seq_len):

                if self.adjacency_type == 0:
                    adj_pred = adjs_fully_connected_pred(seq_start_end).cuda()
                elif self.adjacency_type == 1:
                    adj_pred = adjs_distance_sim_pred(self.sigma, seq_start_end, x_t_abs.detach().cpu()).cuda()
                elif self.adjacency_type == 2:
                    adj_pred = adjs_knn_sim_pred(self.top_k_neigh, seq_start_end, x_t_abs.detach().cpu()).cuda()
                
               
                gat_t = self.gat_enc(x_t_abs.clone(), vae_adj_encod_out)

                phi_pri_x_t = self.phi_x(gat_t)

                # prior

                past_prior_x = pred_traj[:timestep].permute(1, 0, 2).contiguous()
                padding_num = self.obs_len - past_prior_x.shape[1]
                padding_ = torch.zeros(past_prior_x.shape[0], padding_num, past_prior_x.shape[2]).cuda()
                past_prior_x = torch.cat([past_prior_x, padding_], dim=1)

                prior_emb_x = self.prior_emb(past_prior_x)
                prior_emb_x = prior_emb_x.reshape(pred_traj.shape[1], -1)

                hit_combined = self.hit_player_net(hit_t, h[-1])
                shot_combined = self.shot_type_net(s_t, hit_combined,h[-1])
                
                # sampling agents' goals + graph refinement step
                hit_t = hit_combined
                s_t = shot_combined 
                
                hit_pred.append(hit_combined)
                shot_pred.append(shot_combined)

                prior_mean_t, prior_logvar_t = self.prior(torch.cat([prior_emb_x, phi_pri_x_t, shot_combined, hit_combined, h[-1]], 1))

                # sampling from latent
                z_t = self._reparameterize(prior_mean_t, prior_logvar_t)

                phi_z_t = self.phi_z(z_t)
                dec_mean_t, _ = self.dec( phi_z_t)
                samples[timestep] = dec_mean_t

                # feature extraction for reconstructed samples
                phi_x_t = self.phi_x_dec(dec_mean_t)

                # vrnn recurrence
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

                # graph refinement for agents' hiddens
                if self.adjacency_type == 0:
                    adj_pred = adjs_fully_connected_pred(seq_start_end).cuda()
                elif self.adjacency_type == 1:
                    adj_pred = adjs_distance_sim_pred(self.sigma, seq_start_end, x_t_abs.detach().cpu()).cuda()
                elif self.adjacency_type == 2:
                    adj_pred = adjs_knn_sim_pred(self.top_k_neigh, seq_start_end, x_t_abs.detach().cpu()).cuda()

                h_graph = self.graph_hiddens(h[-1].clone(), adj_pred)
                h[-1] = self.history(torch.cat((h_graph, h[-1]), dim=-1)).unsqueeze(0)
                # new abs pos
                x_t_abs = x_t_abs + dec_mean_t
                x_pred_next = x_t_abs
            
                pred_traj[timestep + 1] = x_pred_next

        hit_pred = torch.stack(hit_pred)
        shot_pred = torch.stack(shot_pred)
        
        return samples, hit_pred, shot_pred
