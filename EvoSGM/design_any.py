import torch
from pathlib import Path
from utils import (
    get_model,
    restore_checkpoint,
    get_conditions_random,
    get_mask_all_lengths,
    get_conditions_from_pdb,
)
from ema import ExponentialMovingAverage
import sde_lib as sde_lib
import sampling as sampling
import losses as losses
import pickle as pkl
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
import math
import numpy as np
from pathlib import Path
import pickle as pkl
from pyrosetta import *
import rosetta_min.run as rosetta
import argparse
import time


def get_reverse_mask(seq_length, res_mask) -> str:
    # 初始化掩码列表和当前索引
    opposite_mask = []
    current_index = 1

    # 遍历res_mask中的每个元素
    res_mask = res_mask.split(",")
    for interval in res_mask:
        if ":" in interval:
            start, end = map(int, interval.split(":"))
            # 如果当前索引小于开始索引，则添加区间
            if current_index < start:
                if start - current_index == 1:
                    opposite_mask.append(str(current_index))
                else:
                    opposite_mask.append(f"{current_index}:{start - 1}")
            # 更新当前索引为结束索引的下一个位置
            current_index = end + 1
        else:
            # 对于单个数字，直接处理
            position = int(interval)
            if current_index < position:
                if position - current_index == 1:
                    opposite_mask.append(str(current_index))
                else:
                    opposite_mask.append(f"{current_index}:{position - 1}")
            current_index = position + 1

    # 处理最后一个区间到序列末尾的部分
    if current_index <= seq_length:
        if seq_length - current_index == 0:
            opposite_mask.append(str(current_index))
        else:
            opposite_mask.append(f"{current_index}:{seq_length}")

    return ",".join(opposite_mask)


def sampling_6d(
    config_path: str,
    pdb_path: str,
    cpt_path: str,
    mask_info: str,
    chain: str = "A",
    device: str = "cuda",
    n_iter: int = 10,
    batch_size: int = 16,
    protein_sgm: bool = False,
) -> str:

    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))

    pdb_name = Path(pdb_path).stem
    if protein_sgm:
        workdir = Path("benchmark", "output_bbs", "proteinsgm", pdb_name)
    else:
        workdir = Path("benchmark", "output_bbs", "evosgm", pdb_name)

    # Load Model
    print(f"Loading model from {cpt_path} ...")
    score_model = get_model(config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    state = restore_checkpoint(cpt_path, state, device)
    state["ema"].store(state["model"].parameters())
    state["ema"].copy_to(state["model"].parameters())

    # Load SDE
    if config.training.sde == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    elif config.training.sde == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3

    # Sampling function
    sampling_shape = (
        batch_size,
        config.data.num_channels,
        config.data.max_res_num,
        config.data.max_res_num,
    )
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, sampling_eps)

    select_length = False
    length_index = 1
    print(f"Generating {n_iter * batch_size} 6d_coordinates ...")
    generated_samples = []
    att_dir = "./benchmark/att/" if not protein_sgm else None
    for _ in tqdm(range(n_iter)):
        if select_length:
            mask = get_mask_all_lengths(config, batch_size=batch_size)[length_index - 1]
            condition = {"length": mask.to(config.device)}
        elif pdb_path is not None:
            condition = get_conditions_from_pdb(
                pdb_path, config, att_dir, chain, mask_info, batch_size
            )  # 选定位置的掩码为1
        else:
            condition = get_conditions_random(config, batch_size=batch_size)
        sample, n = sampling_fn(state["model"], condition)
        generated_samples.append(sample.cpu())
    generated_samples = torch.cat(generated_samples, 0)  # 在batch维度拼接成一个大向量
    workdir.mkdir(parents=True, exist_ok=True)

    # 8 min batch_size = 16
    pkl_name = "samples_6d_{}.pkl".format(pdb_name)
    with open(workdir.joinpath(pkl_name), "wb") as f:  # 将生成的样本张量保存到.pkl文件
        pkl.dump(generated_samples, f)
    print(f"Successfully generate 6d_coods in the shape of {generated_samples.shape}!")
    return str(workdir.joinpath(pkl_name))


def rosetta_minimization(
    pdb_path: str,
    pkl_path: str,
    mask_info: str,
    nums=100,
    n_iter=1,  # minimize几条骨架
    fastdesign=False,
    fastrelax=False,
    dist_std=2,
    angle_std=20,
    protein_sgm=False,
):
    tag = Path(pdb_path).stem  # 输出文件夹名

    with open(pkl_path, "rb") as f:
        samples = pkl.load(f)

    def calc_npz(sample):
        msk = np.round(sample[-1])  # 四舍五入
        L = math.sqrt(len(msk[msk == 1]))
        if not (L).is_integer():
            raise ValueError("Terminated due to improper masking channel...")
        else:
            L = int(L)

        npz = {}
        for idx, name in enumerate(["dist", "omega", "theta", "phi"]):
            npz[name] = np.clip(sample[idx][msk == 1].reshape(L, L), -1, 1)
        # Inverse scaling 恢复真实值
        npz["dist_abs"] = (npz["dist"] + 1) * 10
        npz["omega_abs"] = npz["omega"] * math.pi
        npz["theta_abs"] = npz["theta"] * math.pi
        npz["phi_abs"] = (npz["phi"] + 1) * math.pi / 2
        return npz

    rosetta.init_pyrosetta()

    pose = pose_from_pdb(pdb_path)
    seq = pose.sequence()
    pose = None
    # NOTE!!!
    res_mask = mask_info.split(",")
    for r in res_mask:
        if ":" in r:
            # print(r.split(":"))
            start_idx, end_idx = map(int, r.split(":"))
            # print(start_idx,end_idx)
            # seq[int(start_idx) - 1 : int(end_idx) - 1] = "_"
            seq = seq[: start_idx - 1] + "G" * (end_idx - start_idx + 1) + seq[end_idx:]
        else:
            seq = seq[: int(r) - 1] + "G" + seq[int(r) :]
    print(f"Seq to design: {seq}")
    # NOTE: not very elegant
    # msk = np.round(samples[0][-1])
    # L = math.sqrt(len(msk[msk == 1]))
    # seq = "A" * int(L)
    # pose = None
    # print(type(seq))

    ### HARD-CODED FOR PROPER NAMING ### .parent.stem
    if protein_sgm:
        outPath = Path("benchmark", "output_bbs", "proteinsgm", tag)
    else:
        outPath = Path("benchmark", "output_bbs", "evosgm", tag)

    print(
        f"Generating {n_iter} backbones for {nums} 6d_coords each using pyrosetta ..."
    )

    for n in range(n_iter):
        # outPath_run = outPath.joinpath(f"round_{n + 1}")
        for index in tqdm(range(nums)):
            npz = calc_npz(samples[index])
            outPath_run = outPath.joinpath(f"index_{index + 1}")
            if outPath_run.joinpath("final_structure.pdb").is_file():
                continue
            # print("It's OK here in line 225!")
            _ = rosetta.run_minimization(
                npz,
                seq,
                pose=pose,
                scriptdir=Path("rosetta_min"),
                outPath=outPath_run,
                angle_std=angle_std,  # Angular harmonic std
                dist_std=dist_std,  # Distance harmonic std
                use_fastdesign=fastdesign,
                use_fastrelax=fastrelax,
            )

    print(f"Successfully design backbones for {tag}!")
    # # Create symlink
    # score_fn = ScoreFunction()
    # score_fn.add_weights_from_file(
    #     str(Path("rosetta_min").joinpath("data/scorefxn_cart.wts"))
    # )
    # filename = "structure_before_design.pdb"

    # with open(outPath.joinpath("sample.pkl"), "wb") as f:
    #     pkl.dump(sample, f)

    # n_iter10 2min


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_path", type=str)
    parser.add_argument("--chain", type=str)
    parser.add_argument("--mask", type=str)
    parser.add_argument("--length", type=int)
    parser.add_argument("--config_path", type=str)
    parser.add_argument(
        "--cpt_path",
        type=str,
        default="./checkpoints/evosgm/cond_length_inpainting.pth",
    )
    parser.add_argument("--proteinsgm", type=bool, default=False)
    args = parser.parse_args()

    start_time = time.time()

    pdb_path = Path(args.pdb_path)
    pdb_name = pdb_path.stem
    print(
        f"Start designing 6d coords, output directory: ./benchmark/ouput_bbs/{pdb_name}"
    )

    pkl_path = sampling_6d(
        args.config_path,
        args.pdb_path,
        args.cpt_path,
        args.mask,
        args.chain,
        protein_sgm=args.proteinsgm,
        n_iter=4,
        batch_size=32,
    )

    print(f"Start Rosetta minimization!")
    reverse_mask = get_reverse_mask(args.length, args.mask)
    rosetta_minimization(
        args.pdb_path,
        pkl_path,
        reverse_mask,
        protein_sgm=args.proteinsgm,
        nums=100,
    )

    end_time = time.time()
    print(f"Time consumed for designing {pdb_name}: {(end_time-start_time)//60}min!\n")


if __name__ == "__main__":
    main()
