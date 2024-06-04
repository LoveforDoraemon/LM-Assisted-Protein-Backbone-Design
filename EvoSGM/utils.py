# import matplotlib.pyplot as plt
import subprocess
import tempfile

import numpy as np
import torch
import random
from pathlib import Path
from dataset import ProteinDataset, PaddingCollate
from biotite.structure.io import load_structure, save_structure
import biotite.structure as struc
import ncsnpp
import sde_lib


def get_model(config):
    score_model = ncsnpp.NCSNpp(config)
    score_model = score_model.to(config.device)  # 确保移动到指定设备
    score_model = torch.nn.DataParallel(score_model)  # 并行处理
    return score_model


def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(
        ckpt_dir, map_location=device
    )  # 从指定的检查点目录中加载训练状态到state
    state["optimizer"].load_state_dict(loaded_state["optimizer"])
    state["model"].load_state_dict(loaded_state["model"], strict=False)
    state["ema"].load_state_dict(loaded_state["ema"])
    state["step"] = loaded_state["step"]
    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
    }
    torch.save(saved_state, ckpt_dir)  # 将上述以字典保存到.pt，覆盖原有文件ckpt_dir


def recursive_to(obj, device):  # 将不同类型对象转移到device
    if isinstance(obj, torch.Tensor):
        if device == "cpu":
            return obj.cpu()
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}
    else:
        return obj


def random_mask_batch(batch, config):
    if "inpainting" not in config.model.condition:
        batch["mask_inpaint"] = None
        return batch

    B, _, N, _ = batch["coords_6d"].shape
    mask_min = config.model.inpainting.mask_min_len
    mask_max = config.model.inpainting.mask_max_len

    random_mask_prob = config.model.inpainting.random_mask_prob
    contiguous_mask_prob = config.model.inpainting.contiguous_mask_prob

    lengths = [
        len([a for a in i if a != "_"]) for i in batch["aa_str"]
    ]  # get lengths without padding token
    # Decide between none vs random masking vs contiguous masking
    prob = random.random()
    if prob < random_mask_prob:
        # Random masking
        mask = []
        for l in lengths:
            rand = torch.randint(int(mask_min * l), int(mask_max * l), (1,))[0]
            rand_indices = torch.randperm(l)[:rand]

            m = torch.zeros(N)
            m[rand_indices] = 1

            mask.append(m)
        mask = torch.stack(mask, dim=0)
    elif prob > 1 - contiguous_mask_prob:
        # Contiguous masking
        mask = []
        for l in lengths:
            rand = torch.randint(int(mask_min * l), int(mask_max * l), (1,))[0]
            index = torch.randint(0, (l - rand).int(), (1,))[0]

            m = torch.zeros(N)
            m[index : index + rand] = 1

            mask.append(m)
        mask = torch.stack(mask, dim=0)
    else:
        mask = torch.ones(B, N)  # No masking

    mask = torch.logical_or(mask.unsqueeze(-1), mask.unsqueeze(1))  # B, N -> B, N, N
    batch["mask_inpaint"] = mask.to(device=config.device, dtype=torch.bool)

    return batch


def selected_mask_batch(batch, mask_info, config):
    if "inpainting" not in config.model.condition:
        batch["mask_inpaint"] = None
        return batch

    B, _, N, _ = batch["coords_6d"].shape
    mask = torch.zeros(B, N)

    res_mask = mask_info.split(",")
    for r in res_mask:
        if ":" in r:
            start_idx, end_idx = r.split(":")
            mask[:, int(start_idx) - 1 : int(end_idx)] = 1
        else:
            mask[:, int(r) - 1] = 1

    mask = torch.logical_or(mask.unsqueeze(-1), mask.unsqueeze(1))  # B, N -> B, N, N
    # print(f"mask_inapint: {mask[0][0]}")
    batch["mask_inpaint"] = mask.to(device=config.device, dtype=torch.bool)

    return batch


def get_condition_from_batch(config, batch, mask_info=None):
    # print(f"mask_info: {mask_info}!")
    batch_size = batch["coords_6d"].shape[0]  # B,C,N,N
    out = {}
    for c in config.model.condition:
        if c == "length":
            lengths = [
                len([a for a in i if a != "_"]) for i in batch["aa_str"]
            ]  # batch每个蛋白的长度
            mask = torch.zeros(
                batch_size, config.data.max_res_num, config.data.max_res_num
            ).bool()  # B,N-res,N-res 蛋白长度掩码
            for idx, l in enumerate(lengths):
                mask[idx, :l, :l] = True
            out[c] = mask  # bool B,max_res_num,max_res_num
        elif c == "ss":
            out[c] = batch["coords_6d"][:, 5:8]
        elif c == "inpainting":
            if mask_info is not None:
                batch_masked = selected_mask_batch(batch, mask_info, config)
            else:
                batch_masked = random_mask_batch(batch, config)
            out[c] = {
                "coords_6d": batch_masked["coords_6d"],
                "mask_inpaint": batch_masked["mask_inpaint"],
            }

    return recursive_to(out, config.device)


def get_conditions_random(config, batch_size=8):
    # Randomly sample pdbs from dataset
    # Load into dataset/loader and extract info: not very elegant
    paths = list(Path(config.data.dataset_path).iterdir())
    selected = np.random.choice(
        paths, 100, replace=False
    )  # 不重复的从paths中选择100个不同的元素
    ss_constraints = True if config.data.num_channels == 8 else False
    ds = ProteinDataset(
        config.data.dataset_path,
        config.data.attention_path,
        config.data.min_res_num,
        config.data.max_res_num,
        ss_constraints,
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, collate_fn=PaddingCollate(config.data.max_res_num)
    )
    batch = next(iter(dl))
    condition = get_condition_from_batch(config, batch)
    return condition


# 需要修改ProteinDataset参数
def get_conditions_from_pdb(
    pdb, config, att_dir, chain="A", mask_info=None, batch_size=8
):
    tempdir = tempfile.TemporaryDirectory()
    # isolate chain
    st = load_structure(pdb)
    st_chain = st[struc.filter_amino_acids(st) & (st.chain_id == chain)]
    save_structure(
        Path(tempdir.name).joinpath(f"{Path(pdb).stem}_chain_{chain}.pdb"), st_chain
    )

    ss_constraints = True if config.data.num_channels == 9 else False

    ds = ProteinDataset(
        tempdir.name,
        att_dir,
        config.data.min_res_num,
        config.data.max_res_num,
        ss_constraints,
    )
    # print(ds[0]["coords_6d"].shape)
    # print(ds[0]["coords_6d"][-1])

    dl = torch.utils.data.DataLoader(
        [ds[0]] * batch_size,
        batch_size=batch_size,
        collate_fn=PaddingCollate(config.data.max_res_num),
    )
    batch = next(iter(dl))
    # print(batch["coords_6d"].shape)
    # print(batch["coords_6d"][0][-1])

    return get_condition_from_batch(config, batch, mask_info=mask_info)


def get_mask_all_lengths(config, batch_size=16):
    all_lengths = np.arange(config.data.min_res_num, config.data.max_res_num + 1)

    mask = torch.zeros(
        len(all_lengths), batch_size, config.data.max_res_num, config.data.max_res_num
    ).bool()  # L, B, N, N

    for idx, l in enumerate(all_lengths):
        mask[idx, :, :l, :l] = True

    return mask  # 表示不同长度的蛋白质结构 10000 11000 11100 11110 11111


def run_tmalign(path1, path2, binary_path="tm/TMalign", fast=True):
    cmd = [binary_path, path1, path2]
    if fast:
        cmd += ["-fast"]
    result = subprocess.run(cmd, capture_output=True)
    result = result.stdout.decode("UTF-8").split("\n")
    if len(result) < 10:
        return 0.0  # when TMalign throws error
    tm = result[13].split(" ")[1].strip()
    return float(tm)


def show_all_channels(sample, path=None, nrows=1, ncols=8):  # 显示样本所有通道的图像
    from mpl_toolkits.axes_grid1 import ImageGrid
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.1, share_all=True)

    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    ax_idx = 0
    for s in sample:
        for ch in range(ncols):
            grid[ax_idx].imshow(s[ch])
            ax_idx += 1

    if path:
        plt.savefig(path)


_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    # *表示其之后的参数必须使用关键字传递
    # 装饰器函数可以在不修改原始对象(类/函数)代码的情况下对其进行增强或修改
    # 装饰器函数在调用前要加@
    # 嵌套的目的是既可以看作装饰器，又可以看作普通函数
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_MODEL(name):
    return _MODELS[name]


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(
            np.log(config.model.sigma_max),
            np.log(config.model.sigma_min),
            config.model.num_scales,
        )
    )
    # 通过对数线性插值获得一系列噪声水平
    return sigmas


def get_ddpm_params(config):
    """Get betas and alphas --- parameters used in the original DDPM paper."""
    num_diffusion_timesteps = 1000
    # parameters need to be adapted if number of time steps differs from 1000
    beta_start = config.model.beta_min / config.model.num_scales
    beta_end = config.model.beta_max / config.model.num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_1m_alphas_cumprod": sqrt_1m_alphas_cumprod,
        "beta_min": beta_start * (num_diffusion_timesteps - 1),
        "beta_max": beta_end * (num_diffusion_timesteps - 1),
        "num_diffusion_timesteps": num_diffusion_timesteps,
    }


def create_model(config):
    """Create the score model."""
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    score_model = score_model.to(config.device)
    score_model = torch.nn.DataParallel(score_model)
    return score_model


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.
    Args:
      model: The score model.
      train: `True` for training and `False` for evaluation.
    Returns:
      A model function.
    """

    def model_fn(x, labels):
        """Compute the output of the score-based model.
        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.
        Returns:
          A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A score model.
      train: `True` for training and `False` for evaluation.
      continuous: If `True`, the score-based model is expected to directly take continuous time steps.
    Returns:
      A score function.
    """
    model_fn = get_model_fn(model, train=train)

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):

        def score_fn(x, t):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model_fn(x, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None, None, None]
            return score

    elif isinstance(sde, sde_lib.VESDE):

        def score_fn(x, t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()
            score = model_fn(x, labels)
            return score

    else:
        raise NotImplementedError(
            f"SDE class {sde.__class__.__name__} not yet supported."
        )

    return score_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))
