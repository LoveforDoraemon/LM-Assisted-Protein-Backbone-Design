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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="evosgm")
    args = parser.parse_args()

    config_path = args.model + "_length.yml"
    cpt_path = "./checkpoints/" + args.model + "/cond_length_inpainting.pth"
    device = "cuda"
    n_iter = 1
    # 用于随机长度生成的参数
    batch_size = 12

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
    length_indexes = [i for i in range(1, 90)]  # NOTE: 1-90

    generated_samples = []
    for length_index in length_indexes:
        for _ in tqdm(
            range(n_iter)
        ):  # 用tqdm创建迭代器，效果是在args.n_iter(默认1)次迭代的过程中，命令行显示进度条
            if select_length:
                mask = get_mask_all_lengths(config, batch_size=batch_size)[
                    length_index - 1
                ]
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
        workdir.joinpath(args.model + "_adjacent.pkl"), "wb"
    ) as f:  # 将生成的样本张量保存到.pkl文件
        pkl.dump(generated_samples, f)


if __name__ == "__main__":
    main()
