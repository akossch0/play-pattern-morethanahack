import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width: int, kernel: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel // 2)
        self.first = nn.Conv1d(width, width, kernel, dilation=dilation, padding=padding)
        self.second = nn.Conv1d(width, width, kernel, dilation=dilation, padding=padding)
        self.dropout = nn.Dropout(.10)

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        hidden = self.dropout(F.relu(self.first(values)))
        if mask is not None:
            hidden = hidden * mask
        hidden = F.relu(values + self.second(hidden))
        return hidden if mask is None else hidden * mask


class TCN(nn.Module):
    def __init__(self, channels: int, outputs: int) -> None:
        super().__init__()
        width, kernel = 48, 5
        dilations = (1, 2, 4, 8, 16)
        # One-sided receptive field, in input steps: each block has two convs that each reach
        # dilation * (kernel - 1) / 2 steps either side. tcn_inference extends each chunk by up to
        # this many real frames on each side (clamped to the period) and crops the output back.
        self.radius = (kernel - 1) * sum(dilations)
        self.input = nn.Conv1d(channels, width, 1)
        self.blocks = nn.ModuleList(
            ResidualBlock(width, kernel, dilation) for dilation in dilations
        )
        self.output = nn.Conv1d(width, outputs, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        hidden = F.relu(self.input(values))
        if mask is not None:
            hidden = hidden * mask
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.output(hidden)


class TransferTCN(nn.Module):
    def __init__(self, channels: int, outputs: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.temporal = TCN(channels, outputs)
        self.radius = self.temporal.radius

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        views = torch.cat((values[:, :self.channels], values[:, self.channels:]))
        view_mask = None if mask is None else mask.repeat(2, 1, 1)
        home, away = self.temporal(views, view_mask).split(len(values))
        home_team = (home[:, 0] + away[:, 1]) / 2
        away_team = (away[:, 0] + home[:, 1]) / 2
        dead = ((home[:, 2] + away[:, 2]) / 2).unsqueeze(1)
        live = F.log_softmax(torch.stack((home_team, away_team), dim=1), dim=1)
        probability = torch.cat((live + F.logsigmoid(-dead), F.logsigmoid(dead)), dim=1)
        # outputs == 3 (plain TransferTCN): class log-probabilities only.
        # TransitionTCN (outputs=6) also returns the edge logits below.
        if home.shape[1] == 3:
            return probability
        edges = torch.stack((home[:, 3], away[:, 3], home[:, 4], away[:, 4],
                             home[:, 5], away[:, 5]), dim=1)
        return torch.cat((probability, edges), dim=1)


class TransitionTCN(TransferTCN):
    def __init__(self, channels: int) -> None:
        super().__init__(channels, outputs=6)
