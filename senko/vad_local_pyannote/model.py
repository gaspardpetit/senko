# MIT License
#
# Copyright (c) 2020-2023 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This file vendors the minimum model code needed from pyannote.audio to run the
# bundled segmentation-3.0 checkpoint without importing pyannote.audio at runtime.

from functools import lru_cache
from itertools import pairwise
from typing import Optional

from asteroid_filterbanks import Encoder, ParamSincFB
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


def merge_dict(defaults: dict, custom: Optional[dict] = None) -> dict:
    params = dict(defaults)
    if custom is not None:
        params.update(custom)
    return params


def conv1d_num_frames(num_samples: int, kernel_size=5, stride=1, padding=0, dilation=1) -> int:
    return 1 + (num_samples + 2 * padding - dilation * (kernel_size - 1) - 1) // stride


def multi_conv_num_frames(
    num_samples: int,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
) -> int:
    num_frames = num_samples
    for k, s, p, d in zip(kernel_size, stride, padding, dilation):
        num_frames = conv1d_num_frames(
            num_frames,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
        )
    return num_frames


def conv1d_receptive_field_size(num_frames=1, kernel_size=5, stride=1, padding=0, dilation=1):
    effective_kernel_size = 1 + (kernel_size - 1) * dilation
    return effective_kernel_size + (num_frames - 1) * stride - 2 * padding


def multi_conv_receptive_field_size(
    num_frames: int,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
) -> int:
    receptive_field_size = num_frames
    for k, s, p, d in reversed(list(zip(kernel_size, stride, padding, dilation))):
        receptive_field_size = conv1d_receptive_field_size(
            num_frames=receptive_field_size,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
        )
    return receptive_field_size


def conv1d_receptive_field_center(frame=0, kernel_size=5, stride=1, padding=0, dilation=1):
    effective_kernel_size = 1 + (kernel_size - 1) * dilation
    return frame * stride + (effective_kernel_size - 1) // 2 - padding


def multi_conv_receptive_field_center(
    frame: int,
    kernel_size: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
) -> int:
    receptive_field_center = frame
    for k, s, p, d in reversed(list(zip(kernel_size, stride, padding, dilation))):
        receptive_field_center = conv1d_receptive_field_center(
            frame=receptive_field_center,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
        )
    return receptive_field_center


class SincNet(nn.Module):
    def __init__(self, sample_rate: int = 16000, stride: int = 1):
        super().__init__()

        if sample_rate != 16000:
            raise NotImplementedError("SincNet only supports 16kHz audio for now.")

        self.sample_rate = sample_rate
        self.stride = stride

        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)

        self.conv1d = nn.ModuleList()
        self.pool1d = nn.ModuleList()
        self.norm1d = nn.ModuleList()

        self.conv1d.append(
            Encoder(
                ParamSincFB(
                    80,
                    251,
                    stride=self.stride,
                    sample_rate=sample_rate,
                    min_low_hz=50,
                    min_band_hz=50,
                )
            )
        )
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(80, affine=True))

        self.conv1d.append(nn.Conv1d(80, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(60, affine=True))

        self.conv1d.append(nn.Conv1d(60, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(60, affine=True))

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        kernel_size = [251, 3, 5, 3, 5, 3]
        stride = [self.stride, 3, 1, 3, 1, 3]
        padding = [0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1]
        return multi_conv_num_frames(num_samples, kernel_size, stride, padding, dilation)

    def receptive_field_size(self, num_frames: int = 1) -> int:
        kernel_size = [251, 3, 5, 3, 5, 3]
        stride = [self.stride, 3, 1, 3, 1, 3]
        padding = [0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1]
        return multi_conv_receptive_field_size(num_frames, kernel_size, stride, padding, dilation)

    def receptive_field_center(self, frame: int = 0) -> int:
        kernel_size = [251, 3, 5, 3, 5, 3]
        stride = [self.stride, 3, 1, 3, 1, 3]
        padding = [0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1]
        return multi_conv_receptive_field_center(frame, kernel_size, stride, padding, dilation)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        outputs = self.wav_norm1d(waveforms)
        for index, (conv1d, pool1d, norm1d) in enumerate(zip(self.conv1d, self.pool1d, self.norm1d)):
            outputs = conv1d(outputs)
            if index == 0:
                outputs = torch.abs(outputs)
            outputs = F.leaky_relu(norm1d(pool1d(outputs)))
        return outputs


class LocalPyanNet(nn.Module):
    SINCNET_DEFAULTS = {"stride": 10}
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        output_dim: int,
        sincnet: Optional[dict] = None,
        lstm: Optional[dict] = None,
        linear: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
    ):
        super().__init__()
        if num_channels != 1:
            raise ValueError("LocalPyanNet only supports mono audio.")

        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.output_dim = output_dim

        self.sincnet_cfg = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        self.sincnet_cfg["sample_rate"] = sample_rate
        self.lstm_cfg = merge_dict(self.LSTM_DEFAULTS, lstm)
        self.lstm_cfg["batch_first"] = True
        self.linear_cfg = merge_dict(self.LINEAR_DEFAULTS, linear)

        self.sincnet = SincNet(**self.sincnet_cfg)

        monolithic = self.lstm_cfg["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(self.lstm_cfg)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(60, **multi_layer_lstm)
        else:
            num_layers = self.lstm_cfg["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=self.lstm_cfg["dropout"])

            one_layer_lstm = dict(self.lstm_cfg)
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0
            del one_layer_lstm["monolithic"]

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        60 if i == 0 else self.lstm_cfg["hidden_size"] * (2 if self.lstm_cfg["bidirectional"] else 1),
                        **one_layer_lstm,
                    )
                    for i in range(num_layers)
                ]
            )

        if self.linear_cfg["num_layers"] > 0:
            lstm_out_features = self.lstm_cfg["hidden_size"] * (2 if self.lstm_cfg["bidirectional"] else 1)
            self.linear = nn.ModuleList(
                [
                    nn.Linear(in_features, out_features)
                    for in_features, out_features in pairwise(
                        [lstm_out_features] + [self.linear_cfg["hidden_size"]] * self.linear_cfg["num_layers"]
                    )
                ]
            )

        if self.linear_cfg["num_layers"] > 0:
            in_features = self.linear_cfg["hidden_size"]
        else:
            in_features = self.lstm_cfg["hidden_size"] * (2 if self.lstm_cfg["bidirectional"] else 1)

        self.classifier = nn.Linear(in_features, output_dim)
        self.activation = nn.LogSoftmax(dim=-1)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        return self.sincnet.num_frames(num_samples)

    def receptive_field_size(self, num_frames: int = 1) -> int:
        return self.sincnet.receptive_field_size(num_frames=num_frames)

    def receptive_field_center(self, frame: int = 0) -> int:
        return self.sincnet.receptive_field_center(frame=frame)

    @property
    def frame_duration(self) -> float:
        return self.receptive_field_size(num_frames=1) / self.sample_rate

    @property
    def frame_step(self) -> float:
        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = self.receptive_field_size(num_frames=2) - receptive_field_size
        return receptive_field_step / self.sample_rate

    @property
    def frame_start(self) -> float:
        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_start = self.receptive_field_center(frame=0) - (receptive_field_size - 1) / 2
        return receptive_field_start / self.sample_rate

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        outputs = self.sincnet(waveforms)

        if self.lstm_cfg["monolithic"]:
            outputs, _ = self.lstm(rearrange(outputs, "batch feature frame -> batch frame feature"))
        else:
            outputs = rearrange(outputs, "batch feature frame -> batch frame feature")
            for index, lstm in enumerate(self.lstm):
                outputs, _ = lstm(outputs)
                if index + 1 < self.lstm_cfg["num_layers"]:
                    outputs = self.dropout(outputs)

        if self.linear_cfg["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))
