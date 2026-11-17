import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .tcn import ResidualBlock

# Window geometry, in frames. The detector reads a NATIVE-frame window at every STEP-th frame
# (TAPS). The outer HALO frames on each side are context only: span anchors (FINE every 4
# frames, COARSE every 16) lie in the central CORE frames, and span_inference keeps frame
# outputs for those frames only, advancing windows by CORE.
NATIVE, STEP, HALO = 512, 2, 64
CORE = NATIVE - 2 * HALO
TAPS = np.arange(0, NATIVE, STEP)
FINE = np.arange(HALO, NATIVE - HALO, 4)
COARSE = np.arange(HALO, NATIVE - HALO, 16)
POSITIONS = np.r_[FINE, COARSE]
LEVEL = np.r_[np.zeros(len(FINE), int), np.ones(len(COARSE), int)]
ORDER = np.argsort(POSITIONS, kind="stable")
CENTERS = POSITIONS[ORDER]
LEVEL = LEVEL[ORDER]


class Detector(nn.Module):
    def __init__(self, features):
        super().__init__()
        width = 40
        self.input = nn.Conv1d(features, width, 1)
        self.blocks = nn.ModuleList(
            ResidualBlock(width, 3, dilation) for dilation in (1, 2, 4, 8)
        )
        self.frame = nn.Conv1d(width, 3, 1)
        self.fine = nn.Conv1d(width, 6, 1)
        self.coarse = nn.Sequential(
            nn.AvgPool1d(9, stride=1, padding=4, count_include_pad=False),
            nn.Conv1d(width, 6, 1),
        )
        for name, value in (
            ("fine_at", FINE // STEP),
            ("coarse_at", COARSE // STEP),
            ("order", ORDER),
            ("center", CENTERS.astype(np.float32)),
            ("radius", np.where(LEVEL == 0, 18.0, 48.0).astype(np.float32)),
            ("length", np.where(LEVEL == 0, 27.0, 96.0).astype(np.float32)),
        ):
            self.register_buffer(name, torch.from_numpy(value))

    def forward(self, x):
        """Decode one batch of windows.

        x is (batch, taps, features), one row per STEP-th frame of a NATIVE-frame window.
        Returns, with `anchors` meaning the FINE and COARSE centres merged in CENTERS order:
          frame logits (batch, 3, taps), channel first; span_inference softmaxes over dim 1 and
            documents the order as home, away, dead ball;
          anchor class logits (batch, anchors, 3); span_inference softmaxes them and reads
            the first two as the team;
          spans (batch, anchors, 2), start and end in frames from the window start;
          quality logits (batch, anchors); span_inference applies a sigmoid and multiplies
            it into the span score.
        """
        out = F.gelu(self.input(x.transpose(1, 2)))
        for layer in self.blocks:
            out = layer(out, None)
        fine = self.fine(out).index_select(2, self.fine_at)
        coarse = self.coarse(out).index_select(2, self.coarse_at)
        raw = (
            torch.cat((fine, coarse), dim=2).index_select(2, self.order).transpose(1, 2)
        )
        mid = self.center + self.radius * raw[:, :, 3].tanh()
        half = 0.5 * self.length * torch.exp(2.25 * raw[:, :, 4].tanh())
        spans = torch.stack((mid - half, mid + half), dim=-1)
        return self.frame(out), raw[:, :, :3], spans, raw[:, :, 5]
