import pickle
import json
import glob
import argparse
from pathlib import Path
import math
import pytorch_lightning as pl
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional, Tuple

def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


@torch.no_grad()
def cal_polygon_contour(
    pos: Tensor,  # [n_agent, n_step, n_target, 2]
    head: Tensor,  # [n_agent, n_step, n_target]
    width_length: Tensor,  # [n_agent, 1, 1, 2]
) -> Tensor:  # [n_agent, n_step, n_target, 4, 2]
    x, y = pos[..., 0], pos[..., 1]  # [n_agent, n_step, n_target]
    width, length = width_length[..., 0], width_length[..., 1]  # [n_agent, 1 ,1]

    half_cos = 0.5 * head.cos()  # [n_agent, n_step, n_target]
    half_sin = 0.5 * head.sin()  # [n_agent, n_step, n_target]
    length_cos = length * half_cos  # [n_agent, n_step, n_target]
    length_sin = length * half_sin  # [n_agent, n_step, n_target]
    width_cos = width * half_cos  # [n_agent, n_step, n_target]
    width_sin = width * half_sin  # [n_agent, n_step, n_target]

    left_front_x = x + length_cos - width_sin
    left_front_y = y + length_sin + width_cos
    left_front = torch.stack((left_front_x, left_front_y), dim=-1)

    right_front_x = x + length_cos + width_sin
    right_front_y = y + length_sin - width_cos
    right_front = torch.stack((right_front_x, right_front_y), dim=-1)

    right_back_x = x - length_cos + width_sin
    right_back_y = y - length_sin - width_cos
    right_back = torch.stack((right_back_x, right_back_y), dim=-1)

    left_back_x = x - length_cos - width_sin
    left_back_y = y - length_sin + width_cos
    left_back = torch.stack((left_back_x, left_back_y), dim=-1)

    polygon_contour = torch.stack(
        (left_front, right_front, right_back, left_back), dim=-2
    )

    return polygon_contour


def transform_to_global(
    pos_local: Tensor,  # [n_agent, n_step, 2]
    head_local: Optional[Tensor],  # [n_agent, n_step]
    pos_now: Tensor,  # [n_agent, 2]
    head_now: Tensor,  # [n_agent]
) -> Tuple[Tensor, Optional[Tensor]]:
    cos, sin = head_now.cos(), head_now.sin()
    rot_mat = torch.zeros((head_now.shape[0], 2, 2), device=head_now.device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = sin
    rot_mat[:, 1, 0] = -sin
    rot_mat[:, 1, 1] = cos

    pos_global = torch.bmm(pos_local, rot_mat)  # [n_agent, n_step, 2]*[n_agent, 2, 2]
    pos_global = pos_global + pos_now.unsqueeze(1)
    if head_local is None:
        head_global = None
    else:
        head_global = head_local + head_now.unsqueeze(1)
    return pos_global, head_global


def transform_to_local(
    pos_global: Tensor,  # [n_agent, n_step, 2]
    head_global: Optional[Tensor],  # [n_agent, n_step]
    pos_now: Tensor,  # [n_agent, 2]
    head_now: Tensor,  # [n_agent]
) -> Tuple[Tensor, Optional[Tensor]]:
    cos, sin = head_now.cos(), head_now.sin()
    rot_mat = torch.zeros((head_now.shape[0], 2, 2), device=head_now.device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = -sin
    rot_mat[:, 1, 0] = sin
    rot_mat[:, 1, 1] = cos

    pos_local = pos_global - pos_now.unsqueeze(1)
    pos_local = torch.bmm(pos_local, rot_mat)  # [n_agent, n_step, 2]*[n_agent, 2, 2]
    if head_global is None:
        head_local = None
    else:
        head_local = head_global - head_now.unsqueeze(1)
    return pos_local, head_local



def Kdisk_cluster(
    X,  # [n_trajs, 4, 2], bbox of the last point of the segment
    N,  # int
    tol,  # float
    a_pos,  # [n_trajs, 6, 3], the complete segment
    cal_mean_heading=True,
):
    n_total = X.shape[0]
    ret_traj_list = []

    for i in range(N):
        if i == 0:
            choice_index = 0  # always include [0, 0, 0]
        else:
            choice_index = torch.randint(0, X.shape[0], (1,)).item()

        x0 = X[choice_index]

        res_mask = torch.norm(X - x0, dim=-1).mean(-1) > tol
        if cal_mean_heading:
            ret_traj = a_pos[~res_mask].mean(0, keepdim=True)
        else:
            ret_traj = a_pos[[choice_index]]

        X = X[res_mask]
        a_pos = a_pos[res_mask]
        ret_traj_list.append(ret_traj)

        remain = X.shape[0] * 100.0 / n_total
        n_inside = (~res_mask).sum().item()
        print(f"{i=}, {remain=:.2f}%, {n_inside=}")

    # reorder the trajectory according thier traval distance
    ret_traj = torch.cat(ret_traj_list, dim=0)  # 
        
    return ret_traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action token clustering for trajectory tokenization")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the preprocessed dataset directory containing JSON files")
    parser.add_argument("--output", type=str, default="codebook_cache/agent_vocab.pkl",
                        help="Output path for the vocabulary file (default: codebook_cache/agent_vocab.pkl)")
    parser.add_argument("--num_cluster", type=int, default=2048,
                        help="Vocabulary size / number of clusters (default: 2048)")
    parser.add_argument("--n_trajs", type=int, default=2048000,
                        help="Number of trajectory segments to sample (default: 2048000)")
    parser.add_argument("--tol_dist", type=float, default=0.05,
                        help="Tolerance distance for K-disk clustering (default: 0.05)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    args = parser.parse_args()

    # set seed
    pl.seed_everything(args.seed)

    # data
    n_trajs = args.n_trajs
    data_path = args.data_path
    out_file_path = Path(args.output)
    tol_dist = args.tol_dist
    num_cluster = args.num_cluster

    data_file_path = glob.glob(f"{data_path}/*.json")
    trajs = torch.zeros([1, 1, 3], dtype=torch.float32) # ego veh

    # add trajectory
    with tqdm(total=len(data_file_path), desc=f"n_trajs={n_trajs}") as pbar:
        for file in data_file_path:
            with open(file, 'r') as f:
                data = json.load(f)

            trajectory = data['gt_trajectory']
            pos = torch.tensor([[0, 0]])
            head = torch.tensor([0])

            for t in range(len(trajectory)):
                if trajs.shape[0] < n_trajs:
                    next_pos = torch.tensor([trajectory[t][:2]])
                    next_head = torch.tensor([trajectory[t][2]])

                    l_pos, l_head = transform_to_local(
                        pos_global=next_pos.unsqueeze(0),  # [1, 1, 2]
                        head_global=next_head.unsqueeze(0),  # [1, 1]
                        pos_now=pos,  # [1, 2]
                        head_now=head,  # [1]
                    )
                    # print(l_pos)
                    l_head = wrap_angle(l_head)
                    to_add = torch.cat([l_pos, l_head.unsqueeze(-1)], dim=-1)
                    #if not ((trajs - to_add).abs().sum([1, 2]) < 1e-2).any():
                    trajs = torch.cat([trajs, to_add], dim=0)

                    pos = next_pos.clone()
                    head = next_head.clone()

            pbar.update(1)
    
            if trajs.shape[0] == n_trajs:
                break

        print(trajs)

    res = {"token_all": {}}

    # K-disk cluster
    width_length = torch.tensor([2.0, 4.8])
    width_length = width_length.unsqueeze(0)  # [1, 2]

    contour = cal_polygon_contour(
        pos=trajs[:, -1, :2], head=trajs[:, -1, 2], width_length=width_length
    )  # [n_trajs, 4, 2]

    ret_traj = Kdisk_cluster(X=contour, N=num_cluster, tol=tol_dist, a_pos=trajs)
    ret_traj[:, :, -1] = wrap_angle(ret_traj[:, :, -1])

    plt.scatter(ret_traj[:, 0, 0], ret_traj[:, 0, 1], s=5)
    plt.axis('equal')
    
    # Save visualization
    vis_path = out_file_path.with_suffix('.jpg')
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(vis_path)
    print(f"Saved visualization to {vis_path}")

    contour = cal_polygon_contour(
        pos=ret_traj[:, :, :2],  # [N, 6, 2]
        head=ret_traj[:, :, 2],  # [N, 6]
        width_length=width_length.unsqueeze(0),
    )

    res["token_all"]["veh"] = contour.numpy()

    # Save vocabulary
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file_path, "wb") as f:
        pickle.dump(res, f)
    print(f"Saved vocabulary to {out_file_path}")