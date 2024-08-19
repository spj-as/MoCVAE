import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from os.path import join
from typing import Literal


def seq_collate(data):
    (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, obs_goals, pred_goals, obs_hits, pred_hits) = zip(*data)

    batch_size = len(obs_traj)
    obs_seq_len, n_agents, features = obs_traj[0].shape
    pred_seq_len, _, _ = pred_traj[0].shape

    obs_traj = torch.cat(obs_traj, dim=1)
    pred_traj = torch.cat(pred_traj, dim=1)
    obs_traj_rel = torch.cat(obs_traj_rel, dim=1)
    pred_traj_rel = torch.cat(pred_traj_rel, dim=1)
    obs_goals = torch.cat(obs_goals, dim=1)
    pred_goals = torch.cat(pred_goals, dim=1)
    obs_hits = torch.cat(obs_hits, dim=1)
    pred_hits = torch.cat(pred_hits, dim=1)

    # fixed number of agent for every play -> we can manually build seq_start_end
    idxs = list(range(0, (batch_size * n_agents) + n_agents, n_agents))
    seq_start_end = [[start, end] for start, end in zip(idxs[:], idxs[1:])]

    out = [
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        obs_goals,
        pred_goals,
        obs_hits,
        pred_hits,
        torch.tensor(seq_start_end),
    ]
    return tuple(out)


class BadmintonDataset(Dataset):
    """Dataloder for the Basketball trajectories datasets"""

    def __init__(
        self,
        mode: Literal["train", "test"],
        root: str = None,
        n_agents: int = 4,
        obs_len: int = 11,
        pred_len: int = 1,
    ):
        super(BadmintonDataset, self).__init__()
        assert mode in ["train", "test"], "mode must be either train or test"

        self.data_dir = root or join("data", "badminton", "doubles")
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.n_agents = n_agents

        displacements = np.load(join(self.data_dir, mode, "displacements.npy"), allow_pickle=True)
        velocity = np.load(join(self.data_dir, mode, "velocity.npy"), allow_pickle=True)
        goals = np.load(join(self.data_dir, mode, "goals.npy"), allow_pickle=True)
        hits = np.load(join(self.data_dir, mode, "hit.npy"), allow_pickle=True)

        num_seqs = displacements.shape[1] // self.n_agents
        idxs = [idx for idx in range(0, (num_seqs * self.n_agents) + n_agents, n_agents)]
        seq_start_end = [[start, end] for start, end in zip(idxs[:], idxs[1:])]

        self.num_samples = len(seq_start_end)

        self.obs_traj = torch.from_numpy(displacements).type(torch.float)[:obs_len, :, :]
        self.obs_traj_rel = torch.from_numpy(velocity).type(torch.float)[:obs_len, :, :]
        self.obs_goals = torch.from_numpy(goals).type(torch.float)[:obs_len, :]
        self.obs_hits = torch.from_numpy(hits).type(torch.float)[:obs_len, :, :]
        self.pred_traj = torch.from_numpy(displacements).type(torch.float)[obs_len:, :, :]
        self.pred_traj_rel = torch.from_numpy(velocity).type(torch.float)[obs_len:, :, :]
        self.pred_goals = torch.from_numpy(goals).type(torch.float)[obs_len:, :]
        self.pred_hits = torch.from_numpy(hits).type(torch.float)[obs_len:, :, :]
        self.seq_start_end = seq_start_end

    def __len__(self) -> int:
        return self.num_samples

    @property
    def __max_agents__(self) -> int:
        return self.n_agents

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        start, end = self.seq_start_end[idx]

        out = [
            self.obs_traj[:, start:end, :],
            self.pred_traj[:, start:end, :],
            self.obs_traj_rel[:, start:end, :],
            self.pred_traj_rel[:, start:end, :],
            self.obs_goals[:, start:end],
            self.pred_goals[:, start:end],
            self.obs_hits[:, start:end],
            self.pred_hits[:, start:end],
        ]
        return out


def get_badminton_datasets(
    root: str = None,
    n_agents: int = 4,
    obs_len: int = 11,
    pred_len: int = 1,
) -> tuple[BadmintonDataset, BadmintonDataset]:
    train_set = BadmintonDataset(mode="train", root=root, n_agents=n_agents, obs_len=obs_len, pred_len=pred_len)
    test_set = BadmintonDataset(mode="test", root=root, n_agents=n_agents, obs_len=obs_len, pred_len=pred_len)
    return train_set, test_set
