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
"""Layers for defining NCSN++.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import string


# From layers.py


def get_act(config):
    """Get activation functions from the config file."""

    if config.model.nonlinearity.lower() == "elu":
        return nn.ELU()
    elif config.model.nonlinearity.lower() == "relu":
        return nn.ReLU()
    elif config.model.nonlinearity.lower() == "lrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.nonlinearity.lower() == "swish":
        return nn.SiLU()
    else:
        raise NotImplementedError("activation function does not exist!")


# 根据网络层的输入fan_in，输出fan_out规模来缩放权重的方差，可以有效减轻梯度消失和爆炸
# 方差=scale/fan_in(fan_out)，scale是自定义的缩放比例因子
def variance_scaling(
    scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device="cpu"
):
    """Ported from JAX."""

    # 从JAX库中移植
    # 初始化权重以帮助深度网络训练
    # 可以选择生成服从正态分布或均匀分布的weights
    # fans_in/out 进入/出 某一层网络的参数的个数
    def _compute_fans(shape, in_axis=1, out_axis=0):
        # 计算接受域的大小(接受域是该层每个神经元对输入数据的感知区域，这里=height*width)
        # np.prod(shape)计算各维度大小的乘积product
        # shape[in_axis]输入通道数fan_in，也即通道数
        # shape[out_axis]输出通道数fan_out，也即卷积核个数
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode)
            )
        variance = scale / denominator
        if distribution == "normal":
            # *shape可以将(1,2,3)解包成1,2,3
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (
                torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0
            ) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.0):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, "fan_avg", "uniform")


def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1.0, padding=0):
    """1x1 convolution with DDPM initialization."""
    conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


# 该卷积不会改变张量形状H,W
def ddpm_conv3x3(
    in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1
):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


# 时间步嵌入，timestep是一个时间步序列比如[1,2,3]，帮助网络更好地理解和处理时间依赖性
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


# 单下划线表示为(非强制)内部函数
def _einsum(a, b, c, x, y):
    """a,b,c均为字符列表, x,y是张量"""
    einsum_str = "{},{}->{}".format("".join(a), "".join(b), "".join(c))
    return torch.einsum(
        einsum_str, x, y
    )  # 爱因斯坦求和约定，用于简洁表示乘积、点乘、转置、降维


# 计算张量内积，同时保持代码的可读性，将x的最后一个轴和y的第一个轴相乘并求和，如2,3 3,4 -> 2,4
def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(
        string.ascii_lowercase[: len(x.shape)]
    )  # 从"abc..xyz"中截取前(x轴数)个字母表示各个维度
    y_chars = list(string.ascii_lowercase[len(x.shape) : len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


# Net in Net, 这里实际上就是一个全连接层+调整shape
class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(
            default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True
        )  # shape of W: cc
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(
            0, 2, 3, 1
        )  # 对维度进行转置，将通道维度调到最后，0是batch不用动 bhwc
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)  # bchw


# From layerspp.py
conv1x1 = ddpm_conv1x1
conv3x3 = ddpm_conv3x3
NIN = NIN
default_init = default_init


# 通道注意力块，输入输出形状一致
class AttnBlockpp(nn.Module):
    """Channel-wise self-attention block. Modified from DDPM."""

    def __init__(self, channels, skip_rescale=False, init_scale=0.0):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6
        )
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        # sum of product on channel (1->c)
        # w[b, h, w, i, j] = sum(q[b, c, h, w] * k[b, c, i, j] for c in range(channels))
        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


def naive_upsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1))  # 插入两个大小为1的新维度
    x = x.repeat(
        1, 1, 1, factor, 1, factor
    )  # 对第46维度复制factor次，(1,1,1,1,1,1)变为(2,2,2,1,2,1)
    return torch.reshape(
        x, (-1, C, H * factor, W * factor)
    )  # -1表示自动推断该维度的大小，保证元素数量不变


def naive_downsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
    return torch.mean(x, dim=(3, 5))


class Upsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False):
        super().__init__()  # 调用父类nn.Module的初始化方法，使其可以使用Module的基本功能
        out_ch = out_ch if out_ch else in_ch  # 未指定out_ch则与in_ch相同
        if with_conv:
            self.Conv_0 = conv3x3(in_ch, out_ch)

        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        h = F.interpolate(x, (H * 2, W * 2), "nearest")
        # 插值用于估计在已知数据点之间的未知点，这里直接使用最近的源图像像素值
        if self.with_conv:
            h = self.Conv_0(h)

        return h


class Downsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if with_conv:
            self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)

        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1))
            x = self.Conv_0(x)
        else:
            x = F.avg_pool2d(x, 2, stride=2)

        return x


class ResnetBlockDDPMpp(nn.Module):
    """ResBlock adapted from DDPM."""

    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        conv_shortcut=False,
        dropout=0.1,
        skip_rescale=False,
        init_scale=0.0,
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        # 这里实际上是对每个通道进行归一化
        # GN就是把通道分组，组内归一化，N,G,G/C,H,W
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            if (
                conv_shortcut
            ):  # 是否使用卷积快速连接，在连接(skip connenction)是否先卷积再+
                self.Conv_2 = conv3x3(in_ch, out_ch)  # 先卷积保证通道数一致，从而可加
            else:
                self.NIN_0 = NIN(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class ResnetBlockBigGANpp(nn.Module):
    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        up=False,
        down=False,
        dropout=0.1,
        skip_rescale=True,
        init_scale=0.0,
    ):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )
        self.up = up
        self.down = down

        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))

        if self.up:
            h = naive_upsample_2d(h, factor=2)
            x = naive_upsample_2d(x, factor=2)
        elif self.down:
            h = naive_downsample_2d(h, factor=2)
            x = naive_downsample_2d(x, factor=2)

        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)
