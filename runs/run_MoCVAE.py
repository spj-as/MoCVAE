import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from os.path import join
from utils.seed import set_seed
from utils.dataset import get_badminton_datasets, seq_collate
from utils.checkpoint import save_checkpoint
from utils.adjacency_matrix import compute_adjs, compute_adjs_distsim, compute_adjs_knnsim
from utils.eval import relative_to_abs, average_displacement_error, final_displacement_error, mean_square_error
from models.MoCVAE.MoCVAE import MoCVAE
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from time import time
from utils.annealing import KLAnnealer
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def MoCVAE_cli(parser: ArgumentParser) -> ArgumentParser:
    # Required
    parser.add_argument("--name", type=str, required=True, help="current run name")  # fmt: skip

    # Pretrained Options
    parser.add_argument("--position_emb_path", type=str, default="./models/ConvAutoEncoder/weights/best_model.pth", help="path to pretrained position embedding")  # fmt: skip
    # Dataset Options
    parser.add_argument("--root", type=str, default=join("data", "badminton", "doubles"), help="root directory of dataset")  # fmt: skip
    parser.add_argument("--obs_len", type=int, default=10, help="observation length")  # fmt: skip
    parser.add_argument("--pred_len", type=int, default=2, help="prediction length")  # fmt: skip
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")  # fmt: skip

    # Optimizer Options
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')  # fmt: skip
    parser.add_argument("-e", "--epochs", type=int, default=2000, help="number of epochs")  # fmt: skip

    # Model Options
    parser.add_argument("--clip", type=float, default=3.0, help="gradient clip")  # fmt: skip
    parser.add_argument("--s_dim", type=int, default=16, help="number of shot types")  # fmt: skip
    parser.add_argument("--p_dim", type=int, default=4, help="number of players")  # fmt: skip
    parser.add_argument("--h_dim", type=int, default=32, help="hidden dimension")  # fmt: skip
    parser.add_argument("--n_layers", type=int, default=2, help="number of rnn layers")  # fmt: skip
    parser.add_argument("--x_dim", default=2, type=int, help="feature dimension of single agent")  # fmt: skip
    parser.add_argument("--z_dim", default=16, type=int, help="latent dimension")  # fmt: skip
    parser.add_argument("--rnn_dim", default=32, type=int, help="rnn hidden dimension")  # fmt: skip
    parser.add_argument("--resume", action="store_true", default=False, help="resume training from checkpoint")  # fmt: skip
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path")  # fmt: skip
    parser.add_argument("--ce_weight", default=0.01, type=float, required=False, help="Cross-entropy loss weight")

    # Model Options - Graph
    parser.add_argument("--graph_model", type=str, required=True, choices=["gat", "gcn"], help="graph model")  # fmt: skip
    parser.add_argument("--graph_hid", type=int, default=6, help="graph hidden dimension")  # fmt: skip
    parser.add_argument("--sigma", type=float, default=1.2, help="Sigma value for similarity matrix")  # fmt: skip
    parser.add_argument("--adjacency_type", type=int, default=1, choices=[0, 1, 2], help="Type of adjacency matrix:\n0 (fully connected)\n1 (distances similarity)\n2 (knn similarity)")  # fmt: skip
    parser.add_argument("--top_k_neigh", type=int, default=None)

    # Model Options - Graph (GAT)
    parser.add_argument("--n_heads", type=int, default=2, help="number of heads for gat")  # fmt: skip
    parser.add_argument("--alpha", type=float, default=0.2, help="alpha for leaky relu")  # fmt: skip

    # Model Options - Teacher Forcing
    parser.add_argument("--tf_disable", action="store_true", default=False, help="disable teacher forcing")  # fmt: skip
    parser.add_argument("--tf_threshold", type=float, default=1.0, help="initial teacher forcing threshold")  # fmt: skip

    # Kl annealing strategy arguments
    parser.add_argument("--kl_anneal_type", type=str, default="Cyclical", help="")
    parser.add_argument("--kl_anneal_cycle", type=int, default=20, help="")
    parser.add_argument("--kl_anneal_ratio", type=float, default=1, help="")

    # Miscellaneous
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="mode")  # fmt: skip
    parser.add_argument("--seed", type=int, default=12345, help="random seed")  # fmt: skip
    parser.add_argument("--device", type=str, default="cuda", help="device")  # fmt: skip
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")  # fmt: skip
    parser.add_argument("--save_every", type=int, default=100, help="save every")  # fmt: skip
    parser.add_argument("--eval_every", type=int, default=1, help="evaluate every")  # fmt: skip

    parser.add_argument("--results_dir", type=str, default="results", help="results directory")  # fmt: skip
    parser.add_argument("--num_samples", default=20, type=int, help="Number of samples for evaluation")  # fmt: skip
    return parser


def train(
    args: Namespace,
    epoch: int,
    model: MoCVAE,
    loader: DataLoader,
    writer: SummaryWriter,
    optimizer: optim.Optimizer,
    beta: float,
    tf_threshold: float,
):
    start_time = time()

    losses: list[torch.Tensor] = []
    klds: list[torch.Tensor] = []
    nlls: list[torch.Tensor] = []
    ces: list[torch.Tensor] = []
    ce_hit_players: list[torch.Tensor] = []

    ades: list[torch.Tensor] = []
    fdes: list[torch.Tensor] = []
    mses: list[torch.Tensor] = []

    accs: list[torch.Tensor] = []
    precisions: list[torch.Tensor] = []
    recalls: list[torch.Tensor] = []
    model.train()
    for batch_idx, data in enumerate(loader):
        data = [tensor.to(args.device) for tensor in data]
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_rel_gt,
            obs_goals,
            pred_goals_gt,
            obs_hits_ohe,
            pred_hits_gt,
            seq_start_end,
        ) = data

        obs_shot_type_ohe = obs_goals.to(args.device)
        pred_shot_type_gt_ohe = pred_goals_gt.to(args.device)

        if args.adjacency_type == 0:
            adj_out = compute_adjs(args, seq_start_end).to(args.device)
        elif args.adjacency_type == 1:
            adj_out = compute_adjs_distsim(
                args,
                seq_start_end,
                obs_traj.detach().cpu(),
                pred_traj_gt.detach().cpu(),
            ).to(args.device)
        elif args.adjacency_type == 2:
            adj_out = compute_adjs_knnsim(
                args,
                seq_start_end,
                obs_traj.detach().cpu(),
                pred_traj_gt.detach().cpu(),
            ).to(args.device)

        # during training we feed the entire trjs to the model
        all_traj = torch.cat((obs_traj, pred_traj_gt), dim=0)
        all_traj_rel = torch.cat((obs_traj_rel, pred_traj_rel_gt), dim=0)
        all_goals_ohe = torch.cat((obs_shot_type_ohe, pred_shot_type_gt_ohe), dim=0)
        all_hit = torch.cat((obs_hits_ohe, pred_hits_gt), dim=0)

        kld, nll, ce, ce_hit_player, hidden_state, mse_value, ade_value, fde_value = model.forward(
            traj=all_traj,
            traj_rel=all_traj_rel,
            shot_type_ohe=all_goals_ohe,
            hit_player=all_hit,
            seq_start_end=seq_start_end,
            adj_out=adj_out,
            mask=False,
            tf_threshold=tf_threshold,
            obs_len=args.obs_len,
        )
        # Compute loss
        loss = nll + beta * kld +  0.01*ce +  0.01*ce_hit_player
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()


        ades.append(ade_value)
        fdes.append(fde_value)
        mses.append(mse_value)
        losses.append(loss)
        klds.append(kld)
        nlls.append(nll)
        ces.append(ce)
        ce_hit_players.append(ce_hit_player)

    # Compute mean metrics
    avg_loss = torch.sum(torch.stack(losses)) / len(loader)
    avg_kld = torch.sum(torch.stack(klds)) / len(loader)
    avg_nll = torch.sum(torch.stack(nlls)) / len(loader)
    avg_ce = torch.sum(torch.stack(ces)) / len(loader)
    avg_ce_hit_player = torch.sum(torch.stack(ce_hit_players)) / len(loader)
   
    
    avg_ade = torch.sum(torch.stack(ades)) / len(loader)
    avg_fde = torch.sum(torch.stack(fdes)) / len(loader)
    avg_mse = torch.sum(torch.stack(mses)) / len(loader)
    
    # Log metrics
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/kld", avg_kld, epoch)
    writer.add_scalar("train/nll", avg_nll, epoch)
    writer.add_scalar("train/ce", avg_ce, epoch)
    writer.add_scalar("train/ce_hit_player", avg_ce_hit_player, epoch)
    writer.add_scalar("train/ade", avg_ade, epoch)
    writer.add_scalar("train/fde", avg_fde, epoch)
    writer.add_scalar("train/mse", avg_mse, epoch)

    writer.add_scalar("train/beta", beta, epoch)
    writer.add_scalar("train/tf_threshold", tf_threshold, epoch)

    logging.info(f"Epoch {epoch} (Train) | ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f} | MSE: {avg_mse:.4f} | Loss: {avg_loss:.4f} | KLD: {avg_kld:.4f} | NLL: {avg_nll:.4f} | CE: {avg_ce:.4f} | CE Hit: {avg_ce_hit_player:.4f} ")  # fmt: skip
    # logging.info(f"Epoch {epoch} (Train) | Acc: {avg_acc:.4f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} ")  # fmt: skip

    logging.info(f"Epoch {epoch} (Train) | Beta: {beta:.4f} | TF Threshold: {tf_threshold:.4f} ")  # fmt: skip
    logging.info("Epoch [{}], time elapsed: {:3f}".format(epoch, time() - start_time))


@torch.no_grad()
def evaluation(
    args: Namespace,
    writer: SummaryWriter,
    epoch: int,
    model: MoCVAE,
    loader: DataLoader,
    pic_dir: Path,
    beta: float,
    tf_threshold: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Evaluate the model on the test set

    Returns
    -------
    loss: float
        Test loss
    kld : float
        Test KL divergence
    nll : float
        Test Negative log likelihood loss
    mean_ce : float
        Mean cross entropy loss
    mena_ce_hit : float
        Mean cross entropy of hit net
    ade : float
        Average displacement error
    fde : float
        Final displacement error
    mse : float
        Mean squared error
    """
    model.eval()

    losses: list[torch.Tensor] = []
    klds: list[torch.Tensor] = []
    nlls: list[torch.Tensor] = []
    ces: list[torch.Tensor] = []
    ce_hit_players: list[torch.Tensor] = []

    ades: list[torch.Tensor] = []
    fdes: list[torch.Tensor] = []
    mses: list[torch.Tensor] = []

    # path = 'output.txt'
    # f = open(path, 'w')
    for batch_idx, data in enumerate(loader):
        data = [tensor.to(args.device) for tensor in data]
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_rel_gt,
            obs_goals,
            pred_goals_gt,
            obs_hits_ohe,
            pred_hits_gt,
            seq_start_end,
        ) = data

        obs_shot_type_ohe = obs_goals.to(args.device)
        pred_goals_gt_ohe = pred_goals_gt.to(args.device)

        if args.adjacency_type == 0:
            adj_out = compute_adjs(args, seq_start_end).to(args.device)
        elif args.adjacency_type == 1:
            adj_out = compute_adjs_distsim(
                args,
                seq_start_end,
                obs_traj.detach().cpu(),
                pred_traj_gt.detach().cpu(),
            ).to(args.device)
        elif args.adjacency_type == 2:
            adj_out = compute_adjs_knnsim(
                args,
                seq_start_end,
                obs_traj.detach().cpu(),
                pred_traj_gt.detach().cpu(),
            ).to(args.device)

        all_traj = torch.cat((obs_traj, pred_traj_gt), dim=0)
        all_traj_rel = torch.cat((obs_traj_rel, pred_traj_rel_gt), dim=0)
        all_goals_ohe = torch.cat((obs_shot_type_ohe, pred_goals_gt_ohe), dim=0)
        all_hit = torch.cat((obs_hits_ohe, pred_hits_gt), dim=0)

        (kld, nll, ce, ce_hit_player, hidden_state, mse_value, ade_value, fde_value) = model.forward(
            traj=all_traj,
            traj_rel=all_traj_rel,
            shot_type_ohe=all_goals_ohe,
            hit_player=all_hit,
            seq_start_end=seq_start_end,
            adj_out=adj_out,
            mask=False,
            tf_threshold=tf_threshold,
            obs_len=args.obs_len,
        )
       
        min_ade = float("Inf")
        min_fde = None
        min_mse = None
        min_plot_GT = None
        min_plot_pred = None
        min_acc = float("Inf")
        min_precision = None
        min_recall = None
        
        for _ in range(args.num_samples):
            
            samples_rel, hit_pred, shot_pred = model.sample(
                samples_seq_len=args.pred_len,
                h=hidden_state,
                x_abs_start=obs_traj[-1],
                shot_type_start=obs_shot_type_ohe[-1],
                hit_start=obs_hits_ohe[-1],
                seq_start_end=seq_start_end,
            )

            shot_cal = pred_goals_gt_ohe[:,::4,:].cpu()
            shot_pred_cal = shot_pred[:,::4,:].cpu()
            hit_cal = pred_hits_gt[:,::4,:].cpu()
            hit_pred_cal = hit_pred[:,::4,:].cpu()
            cross_entropy=0
            cross_entropy_hit=0
            for i in range(hit_cal.shape[0]):
                for j in range (hit_cal.shape[1]):
                    cross_entropy_hit+=F.binary_cross_entropy(hit_pred_cal[i][j],hit_cal[i][j] ,reduction="sum")
            for i in range(shot_pred_cal.shape[0]):
                for j in range (shot_pred_cal.shape[1]):
                    cross_entropy+=F.binary_cross_entropy(shot_pred_cal[i][j],shot_cal[i][j] ,reduction="sum")
            cross_entropy  /= (shot_cal.shape[0]*shot_cal.shape[1])
            cross_entropy_hit /=  (hit_cal.shape[0]*hit_cal.shape[1])
         
            
            
            samples = relative_to_abs(samples_rel, obs_traj[-1])
            plot_GT = pred_traj_gt
            plot_pred = samples
      
            ade_ = average_displacement_error(pred=samples, actual=pred_traj_gt, pred_len=args.pred_len)
            fde_ = final_displacement_error(pred=samples, actual=pred_traj_gt, pred_len=args.pred_len)
            mse_ = mean_square_error(pred=samples, actual=pred_traj_gt, pred_len=args.pred_len)

            if ade_ < min_ade:
                min_ade = ade_
                min_fde = fde_
                min_mse = mse_
               
                min_ce= cross_entropy
                min_ce_hit= cross_entropy_hit


        ades.append(min_ade)
        fdes.append(min_fde)
        mses.append(min_mse)

   
        ces.append(min_ce)
        ce_hit_players.append(min_ce_hit)
        
    avg_ce = torch.sum(torch.stack(ces)) / len(loader)
    avg_ce_hit_player = torch.sum(torch.stack(ce_hit_players)) / len(loader)
    
    avg_ade = torch.sum(torch.stack(ades)) / len(loader)
    avg_fde = torch.sum(torch.stack(fdes)) / len(loader)
    avg_mse = torch.sum(torch.stack(mses)) / len(loader)
    
   
    if writer != None:
        writer.add_scalar("test/ce", avg_ce, epoch)
        writer.add_scalar("test/ce_hit_player", avg_ce_hit_player, epoch)
        writer.add_scalar("test/ade", avg_ade, epoch)
        writer.add_scalar("test/fde", avg_fde, epoch)
        writer.add_scalar("test/mse", avg_mse, epoch)

    return  avg_ce, avg_ce_hit_player, avg_ade, avg_fde, avg_mse


def MoCVAE_main(args: Namespace):
    annealer = KLAnnealer(args)
    g = set_seed(args.seed)

    base = Path(args.results_dir, args.name)
    sw_dir = Path(base, "events")
    log_dir = Path(base, "logs")
    ckpt_dir = Path(base, "checkpoints")
    pic_dir = Path(base, "pics")

    sw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pic_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_dir.joinpath(f"{time()}.log"),
        filemode="a",
        level=logging.INFO,
        format="[%(levelname)s | %(asctime)s]: %(message)s",
        datefmt="%Y/%m/%d %H:%M",
    )

    train_set, test_set = get_badminton_datasets(
        root=args.root, n_agents=4, obs_len=args.obs_len, pred_len=args.pred_len
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=seq_collate, num_workers=args.num_workers, generator=g)  # fmt: skip
    test_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=seq_collate, num_workers=args.num_workers, generator=g)  # fmt: skip

    n_max_agents = max(train_set.__max_agents__, test_set.__max_agents__)
    tf_threshold = args.tf_threshold
    if args.mode == "train":
        writer = SummaryWriter(str(sw_dir))
        checkpoint: dict[str] = torch.load(args.checkpoint) if args.resume else {}

        model = MoCVAE(args=args, n_max_player=n_max_agents)
        model.load_state_dict(checkpoint["model"]) if args.resume else None
        model = model.to(args.device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint["optimizer"]) if args.resume else None

        start_epoch = checkpoint["epoch"] if args.resume else 0
        best_ade = checkpoint.get("metrics", {}).get("best_ade") if args.resume else float("Inf")

        for epoch in range(start_epoch + 1, args.epochs + 1):
            # Train the model
            beta = annealer.get_beta()

            train(
                args,
                epoch=epoch,
                model=model,
                loader=train_loader,
                writer=writer,
                optimizer=optimizer,
                beta=beta,
                tf_threshold=tf_threshold,
            )

            # Preriodically evaluate the model
            if epoch % args.eval_every == 0:
                mean_ce, mean_ce_hit, ade, fde, mse = evaluation(
                    args,
                    writer=writer,
                    epoch=epoch,
                    model=model,
                    loader=test_loader,
                    pic_dir=pic_dir,
                    beta=beta,
                    tf_threshold=tf_threshold,
                )
                logging.info(f"Epoch {epoch} (Val) | ADE: {ade:.4f} | FDE: {fde:.4f} | MSE: {mse:.4f} |  CE: {mean_ce:.4f} | CE Hit: {mean_ce_hit:.4f}")  # fmt: skip
                # logging.info(f"Epoch {epoch} (Val) | ACC: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")  # fmt: skip

                if 0 <= ade < best_ade:
                    best_ade = ade
                    path = ckpt_dir.joinpath(f"best_ade_{best_ade:.4f}_epoch_{epoch}.pth")
                    save_checkpoint(str(path), args, epoch, model, optimizer, {"best_ade": best_ade})

            # Periodically save the model
            if epoch % args.save_every == 0:
                path = ckpt_dir.joinpath(f"epoch_{epoch}.pth")
                save_checkpoint(str(path), args, epoch, model, optimizer, {"best_ade": best_ade})
            tf_threshold *= 0.95
            annealer.update()

        writer.close()

    elif args.mode == "eval":
        assert args.checkpoint is not None, "checkpoint path is required for testing"
        evaluation(args)
        checkpoint: dict[str] = torch.load(args.checkpoint)
        
        model = MoCVAE(args=args, n_max_player=n_max_agents)
        model.load_state_dict(checkpoint["model"])
        model = model.to(args.device)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint["optimizer"])

        ade, fde, mse, ce_shot, ce_hit = evaluation(
            args,
            writer=None,  
            epoch=checkpoint["epoch"],
            model=model,
            loader=test_loader,
            pic_dir=pic_dir,
            beta=1.0,  
            tf_threshold=1.0  
        )

        print(f"Evaluation | ADE: {ade:.4f} | FDE: {fde:.4f} | MSE: {mse:.4f} | CE shot: {ce_shot:.4f} | CE hit: {ce_hit:.4f}")  # fmt: skip
