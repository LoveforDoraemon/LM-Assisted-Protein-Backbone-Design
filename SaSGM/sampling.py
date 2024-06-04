# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import abc

from utils import get_score_fn
import sde_lib

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):  # NOTE: name必须用关键字参数传入方式
    """A decorator for registering predictor classes."""

    def _register(cls):  # NOTE: 私有函数
        if name is None:
            local_name = cls.__name__  # 类的名称
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, eps):
    """Create a sampling function.

    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers representing the expected shape of a single sample.
      eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """
    predictor = get_predictor(config.sampling.predictor.lower())  # ReverseDiffusion
    corrector = get_corrector(config.sampling.corrector.lower())  # Langevin
    sampling_fn = get_pc_sampler(
        sde=sde,
        shape=shape,  # B,N,max-res,max-res
        predictor=predictor,
        corrector=corrector,
        snr=config.sampling.snr,
        n_steps=config.sampling.n_steps_each,
        probability_flow=config.sampling.probability_flow,
        denoise=config.sampling.noise_removal,
        eps=eps,
        device=config.device,
    )

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


# 装饰器：在类A定义后立即执行 A = decorator[()](A)
@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, context):
        f, G = self.rsde.discretize(x, t, context)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


@register_corrector(name="langevin")
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t, context):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t, context)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


def shared_predictor_update_fn(x, t, context, sde, model, predictor, probability_flow):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = get_score_fn(sde, model, train=False)
    predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t, context)  # NOTE: context added here


def shared_corrector_update_fn(x, t, context, sde, model, corrector, snr, n_steps):
    """A wrapper that configures and returns the update function of correctors."""
    score_fn = get_score_fn(sde, model, train=False)
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t, context)  # NOTE: context added here


def get_pc_sampler(
    sde,
    shape,
    predictor,
    corrector,
    snr,
    n_steps=1,
    probability_flow=False,
    denoise=True,
    eps=1e-3,
    device="cuda",
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(  # 部分函数
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        snr=snr,
        n_steps=n_steps,
    )

    # NOTE: key sampling algo
    def pc_sampler(model, condition):
        """The PC sampler funciton.

        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            # Conditional information
            conditional_mask = torch.ones_like(x).bool()  # B, C, N, N
            if condition is not None:
                for k, v in condition.items():
                    if k == "length":
                        x = x * v.unsqueeze(1)
                        conditional_mask = conditional_mask * v.unsqueeze(1)
                        x[:, -1] = v
                        conditional_mask[:, -1] = True  # NOTE
                    elif k == "ss":
                        x[:, 5:8] = v
                        conditional_mask[:, 5:8] = True
                    elif k == "inpainting":
                        coords_6d = v["coords_6d"]
                        mask_inpaint = v[
                            "mask_inpaint"
                        ]  # True for regions to be inpainted
                        conditional_mask = conditional_mask * mask_inpaint.unsqueeze(1)
                        conditional_mask[:, -1] = True  # NOTE
                        x = torch.where(conditional_mask, coords_6d, x)
                        # NOTE: add codes to mask embeddings here
                    elif k == "context":
                        # NOTE: code to cope with context here
                        context = v

            x_initial = x.detach().clone()

            for i in range(sde.N):  # default = 1000
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, context=context, model=model)
                x = torch.where(conditional_mask, x_initial, x).float()
                x, x_mean = predictor_update_fn(x, vec_t, context=context, model=model)
                x = torch.where(conditional_mask, x_initial, x).float()
                # if i == 100:
                #     print("100")
                # if i == 500:
                #     print("500")

            x_mean = torch.where(conditional_mask, x_initial, x_mean).float()

            return x_mean if denoise else x, sde.N * (n_steps + 1)

    return pc_sampler