import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns

def one_hot_encode(inds, N):
    dims = [inds.size(i) for i in range(len(inds.size()))]
    inds = inds.unsqueeze(-1).long()
    dims.append(N)
    ret = (torch.zeros(dims)).cuda()
    ret.scatter_(-1, inds, 1)
    return ret


def sample_multinomial(probs):
    """Each element of probs tensor [shape = (batch, s_dim)] has 's_dim' probabilities (one for each grid cell),
    i.e. is a row containing a probability distribution for the goal. We sample n (=batch) indices (one for each row)
    from these distributions, and covert it to a 1-hot encoding."""
    # probs = probs / probs.sum(dim=1, keepdim=True)
    assert (probs >= 0).all(), "Negative probabilities found"
    assert not torch.isnan(probs).any(), "NaN values found"
    assert not torch.isinf(probs).any(), "Infinite values found"
    inds = torch.multinomial(probs, 1).data.long().squeeze()
    ret = one_hot_encode(inds, probs.size(-1))
    return ret


def average_displacement_error(
    pred: torch.Tensor,
    actual: torch.Tensor,
    pred_len: int = 1,
    player_num: int = 4,
) -> torch.Tensor:
    """
    Average Displacement Error

    Parameters
    ----------
    pred : torch.Tensor
        [batch_size, seq_len * player_num, input_size]
    actual : torch.Tensor
        [batch_size, seq_len * player_num, input_size]

    Returns
    -------
    torch.Tensor
        Return loss
    """
    pred_ = pred.permute(1, 0, 2).reshape(-1, 4, pred_len, 2).permute(0, 2, 1, 3)
    actual_ = actual.permute(1, 0, 2).reshape(-1, 4, pred_len, 2).permute(0, 2, 1, 3)

    batch_size, pred_len, _, _ = pred_.shape  # pred_ shape (batch_size, pred_len, 4, 2)

    total_ADE = 0.0
    for batch in range(batch_size):
        ade_numerator = 0.0
        ade_denominator = 0

        for step in range(pred_len):
            mask_val = (actual_[batch, step, 0, 0] != 0) and (actual_[batch, step, 0, 1] != 0)
            if mask_val:
                ade_denominator += 1

                err_step = (pred_[batch, step, :, :] - actual_[batch, step, :, :]) ** 2
                err_sqrt_sum = torch.sum(torch.sqrt(torch.sum(err_step, dim=-1)), dim=-1)
                if err_sqrt_sum != err_sqrt_sum:
                    err_sqrt_sum = 0.0

                ade_numerator += err_sqrt_sum
        if ade_denominator > 0:
            ade_batch = ade_numerator / float(ade_denominator)
            total_ADE += ade_batch
    if batch_size > 0:
        ADE = total_ADE / batch_size
        ADE = ADE / player_num

    # print("Average Displacement Error:", ADE)

    return ADE


def final_displacement_error(
    pred: torch.Tensor,
    actual: torch.Tensor,
    player_num: int = 4,
    pred_len: int = 1,
) -> torch.Tensor:
    """
    Final Displacement Error

    Parameters
    ----------
    pred : torch.Tensor
        [batch_size, seq_len, player_num, input_size]
    actual : torch.Tensor
        [batch_size, seq_len, player_num, input_size]
    player_num : int, optional
        Number of players, by default 4

    Returns
    -------
    torch.Tensor
        Return loss and batch_size
    """
    pred_ = pred.permute(1, 0, 2).reshape(-1, player_num, pred_len, 2).permute(0, 2, 1, 3)
    actual_ = actual.permute(1, 0, 2).reshape(-1, player_num, pred_len, 2).permute(0, 2, 1, 3)

    batch_size, pred_len, num_players, _ = pred_.shape  # pred_ shape (batch_size, pred_len, num_players, 2)
    total_FDE = 0.0
    for batch in range(batch_size):
        fde_numerator = 0.0
        for player in range(num_players):
            last_idx = -1
            for step in range(pred_len):
                if (actual_[batch, step, player, 0] != 0) and (actual_[batch, step, player, 1] != 0):
                    last_idx = step

            if last_idx == -1:
                continue

            err_last_step = torch.sum((pred_[batch, last_idx, player, :] - actual_[batch, last_idx, player, :]) ** 2)

            fde_sqrt = torch.sqrt(err_last_step)
            if fde_sqrt != fde_sqrt:
                fde_sqrt = 0.0

            fde_numerator += fde_sqrt
        total_FDE += fde_numerator

    if batch_size > 0:
        FDE = total_FDE / batch_size
        FDE /= num_players

    return FDE


def mean_square_error(
    pred: torch.Tensor,
    actual: torch.Tensor,
    pred_len: int = 1,
    player_num: int = 4,
) -> torch.Tensor:
    pred_ = pred.permute(1, 0, 2).reshape(-1, 4, pred_len, 2).permute(0, 2, 1, 3)
    actual_ = actual.permute(1, 0, 2).reshape(-1, 4, pred_len, 2).permute(0, 2, 1, 3)

    batch_size, num_steps, _, _ = actual_.shape
    MSE = 0.0

    for batch in range(batch_size):
        mse_denominator = 0.0
        mse_numerator = 0.0
        for step in range(num_steps):
            mask_val = (actual_[batch, step, 0, 0] != 0) and (actual_[batch, step, 0, 1] != 0)
            if mask_val:
                mse_denominator += 1.0
                err = (pred_[batch, step, :, :] - actual_[batch, step, :, :]) ** 2
                err_sum = torch.sum(err, dim=-1)
                err_sum = torch.sum(err_sum, dim=-1)
                if err_sum != err_sum:
                    err_sum = 0.0
                mse_numerator += err_sum
        if mse_denominator > 0:
            mse_value = mse_numerator / mse_denominator
            MSE += mse_value

    if batch_size > 0:
        MSE = MSE / batch_size
        MSE /= player_num

    return MSE


def relative_to_abs(
    traj_rel: torch.Tensor,
    start_pos: torch.Tensor,
) -> torch.Tensor:
    """
    Inputs:
    - rel_traj: tensor of shape (seq_len, batch, 2), trajectory composed by displacements
    - start_pos: tensor of shape (batch, 2), initial position
    Outputs:
    - input tensor (seq_len, batch, 2) filled with absolute coords
    """
    rel_traj = traj_rel.permute(1, 0, 2)  # (seq_len, batch, 2) -> (batch, seq_len, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos

    return abs_traj.permute(1, 0, 2)
