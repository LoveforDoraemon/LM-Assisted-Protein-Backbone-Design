from pathlib import Path
import losses as losses
import sampling as sampling
from ema import ExponentialMovingAverage
import sde_lib as sde_lib
import torch

from torch.utils import tensorboard
from utils import (
    save_checkpoint,
    restore_checkpoint,
    get_model,
    recursive_to,
    random_mask_batch,
    get_condition_from_batch,
)
from dataset import ProteinDataset, PaddingCollate
import pickle as pkl
import yaml
from easydict import EasyDict


def main():
    with open("sasgm_inpainting.yml", "r") as f:
        config = EasyDict(yaml.safe_load(f))

    dataset = ProteinDataset(
        config.data.dataset_path,
        config.data.min_res_num,
        config.data.max_res_num,
        context_path=config.data.context_path,
    )
    # print(dataset[0][1].device)
    print(f"Finish loading dataset with the size of {len(dataset)}!")

    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_sampler = torch.utils.data.RandomSampler(
        train_ds,
        replacement=True,  # 有放回抽样
        num_samples=config.training.n_iters * config.training.batch_size,
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=config.training.batch_size,
        collate_fn=PaddingCollate(
            config.data.max_res_num, use_context=True
        ),  # 如何将多个样本拼接成一个batch，将张量填充到batch统一的大小
    )
    train_iter = iter(train_dl)  # 之后可以用next来遍历
    test_sampler = torch.utils.data.RandomSampler(
        test_ds,
        replacement=True,
        num_samples=config.training.n_iters * config.training.batch_size,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        sampler=test_sampler,
        batch_size=config.training.batch_size,
        collate_fn=PaddingCollate(config.data.max_res_num, use_context=True),
    )
    test_iter = iter(test_dl)
    print("Finish spliting the train_set and test_set with the ratio of 95:5!")

    # Create directories for experimental logs
    # if args.resume is not None:
    #     workdir = Path(args.resume)
    # else:
    workdir = Path(
        "training",
        "cond_inpainting",
        # "test",
        # "2024_04_23",  # time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime()),
    )
    workdir.mkdir(
        exist_ok=True, parents=True
    )  # exits_ok表示已存在不报错，parents表示同时创建父目录
    # Save config to workdir
    # shutil.copy(args.config, workdir.joinpath("config.yml"))
    sample_dir = workdir.joinpath("samples")  # samples作为workdir的子目录
    sample_dir.mkdir(exist_ok=True)
    tb_dir = workdir.joinpath("tensorboard")
    tb_dir.mkdir(exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = get_model(config)
    # print(score_model)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    optimizer = losses.get_optimizer(config, score_model.parameters())
    # 保存训练信息(优化器，模型，ema，step)的字典
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    # 创建检查点目录
    checkpoint_dir = workdir.joinpath("checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = workdir.joinpath("checkpoints-meta", "checkpoint.pth")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_meta_dir.parent.mkdir(exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    if checkpoint_meta_dir.is_file():
        state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
        initial_step = int(state["step"])
    else:
        initial_step = 0
    print(f"Starting from step {initial_step} ...")

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn)
    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (
            config.training.batch_size,
            config.data.num_channels,
            config.data.max_res_num,
            config.data.max_res_num,
        )
        sampling_fn = sampling.get_sampling_fn(
            config, sde, sampling_shape, sampling_eps
        )

    for step in range(initial_step, config.training.n_iters + 1):
        batch, context_batch = recursive_to(next(train_iter), config.device)
        # print(f"The training batch is a {type(batch)} !")
        # Execute one training step
        batch = random_mask_batch(batch, config)  # mask(opt)
        loss = train_step_fn(
            state, batch, condition=config.model.condition, context=context_batch
        )
        # 保存训练loss
        if step % config.training.log_freq == 0:
            writer.add_scalar("training_loss", loss, step)
        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)
        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            eval_batch, eval_context_batch = recursive_to(
                next(test_iter), config.device
            )
            # print(f"The evaluation batch is a {type(eval_batch)} !")
            # print(len(eval_batch))
            # print(eval_batch)
            eval_batch = random_mask_batch(eval_batch, config)
            eval_loss = eval_step_fn(
                state,
                eval_batch,
                condition=config.model.condition,
                context=eval_context_batch,
            )
            writer.add_scalar("eval_loss", eval_loss.item(), step)
            print(f"Current step is {step} with the eval_loss of {eval_loss.item()}.")
        # Save a checkpoint periodically and generate samples if needed
        if (
            step != 0
            and step % config.training.snapshot_freq == 0
            or step == config.training.n_iters
        ):
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(
                checkpoint_dir.joinpath(f"checkpoint_{save_step}.pth"), state
            )
            # Generate and save samples
            if config.training.snapshot_sampling:  # default=False
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                condition = get_condition_from_batch(config, eval_batch)
                sample, n = sampling_fn(
                    score_model, condition=condition, context=None
                )  # NOTE
                ema.restore(score_model.parameters())
                this_sample_dir = sample_dir.joinpath(f"iter_{step}")
                this_sample_dir.mkdir(exist_ok=True)
                with open(str(this_sample_dir.joinpath("sample.pkl")), "wb") as fout:
                    pkl.dump(sample.cpu(), fout)
                # save_grid(sample.cpu().numpy(), this_sample_dir.joinpath("sample.png"))


if __name__ == "__main__":
    main()
