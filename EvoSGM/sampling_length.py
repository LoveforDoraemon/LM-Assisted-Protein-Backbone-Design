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
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
import argparse
import math
import numpy as np
from pyrosetta import *
import rosetta_min.run as rosetta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="evosgm")
    parser.add_argument("--length", type=int, default=50)
    args = parser.parse_args()

    config_path = args.model + "_length.yml"
    cpt_path = "./checkpoints/" + args.model + "/cond_length.pth"
    device = "cuda"
    n_iter = 4  # 4
    # 用于随机长度生成的参数
    batch_size = 32  # 32

    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))  # 使用.符号可以方便访问字典值的类

    workdir = Path(
        "benchmark",
        "unconditional",
    )

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

    # 用于条件生成的参数
    pdb = None
    chain = "A"
    mask_info = None

    select_length = True
    length_index = args.length - 39  # NOTE: 1-90

    generated_samples = []
    for _ in tqdm(
        range(n_iter)
    ):  # 用tqdm创建迭代器，效果是在args.n_iter(默认1)次迭代的过程中，命令行显示进度条
        if select_length:
            mask = get_mask_all_lengths(config, batch_size=batch_size)[length_index - 1]
            # print(mask)
            condition = {"length": mask.to(config.device)}
        elif pdb is not None:
            condition = get_conditions_from_pdb(
                pdb, config, chain, mask_info, batch_size=batch_size
            )
        else:
            condition = get_conditions_random(config, batch_size=batch_size)
        sample, n = sampling_fn(state["model"], condition)
        generated_samples.append(sample.cpu())

    generated_samples = torch.cat(generated_samples, 0)  # 在batch维度拼接成一个大向量

    workdir.mkdir(parents=True, exist_ok=True)

    # 1min1个

    print(generated_samples.shape)

    with open(
        workdir.joinpath(args.model + "_length_" + str(args.length) + ".pkl"), "wb"
    ) as f:  # 将生成的样本张量保存到.pkl文件
        pkl.dump(generated_samples, f)

    print("Start rosetta minimization!")
    samples = generated_samples
    n_iter = 1
    nums = 100
    fastdesign = False
    fastrelax = False
    dist_std = 2
    angle_std = 20

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

    L = args.length
    seq = "A" * L
    pose = None

    ### HARD-CODED FOR PROPER NAMING ### .parent.stem
    outPath = Path(
        "benchmark", "unconditional", args.model, "length_" + str(L) + "_bbs"
    )

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

    print(f"Successfully design backbones using {args.model} with the length of {L}!")


if __name__ == "__main__":
    main()
