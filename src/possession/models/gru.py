"""Tactics GRU with DeepSets pooling and separate possession and dead-ball heads."""

import torch
from torch import nn

from .gru_features import N_FEAT
from .team_shape import N_EXT, N_GLOB


class SetEncoder(nn.Module):
    def __init__(self, n_in, width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_in, width), nn.GELU(), nn.Linear(width, width), nn.GELU())

    def forward(self, tokens, present):
        h = self.mlp(tokens)
        m = present.unsqueeze(-1).to(h.dtype)
        mean = (h * m).sum(2) / m.sum(2).clamp(min=1.0)
        mx = torch.where(m > 0, h, torch.full_like(h, -1e4)).amax(2)
        mx = torch.where(present.any(2, keepdim=True), mx, torch.zeros_like(mx))
        return torch.cat([mean, mx], -1)


class TacticNet(nn.Module):
    def __init__(self, width=96, hidden=128, layers=2, dropout=0.1, frame=192, styles=1, n_ball=4):
        super().__init__()
        self.n_ball = n_ball                            # 4, or 4 + features.N_TOUCH with touch features
        n_tok = N_FEAT + N_EXT + 3                      # + ball distance, closeness, ball flag
        self.own = SetEncoder(n_tok, width)
        self.opp = SetEncoder(n_tok, width)
        self.frame = nn.Sequential(nn.Linear(4 * width + 2 + N_GLOB + n_ball + 1, frame), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(frame, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        head = lambda n: nn.Sequential(nn.Linear(2 * hidden, 64), nn.GELU(), nn.Linear(64, n))
        # One output per label style (style 0 is the ranked one, DFL). The trunk is shared, so
        # sources whose labels follow another convention still teach it, without bending the
        # head that is used at prediction time.
        self.styles = styles
        self.poss, self.dead, self.bound, self.aux = head(styles), head(styles), head(styles), head(2)

    @staticmethod
    def _with_ball(tok, ball, ball_pres):
        """Append each player's distance to the ball (scaled), a closeness score and the ball flag."""
        # tok[..., :2] and ball[..., :2] are in half-pitch units (gru_features: (x - 0.5) * 2), so
        # 52.5 and 34.0 turn the offsets into metres on a nominal 105 x 68 m pitch.
        dx = (tok[..., 0] - ball[..., None, 0]) * 52.5
        dy = (tok[..., 1] - ball[..., None, 1]) * 34.0
        d = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        f = ball_pres[..., None].to(tok.dtype)
        extra = torch.stack([(d / 20).clamp(max=3) * f, torch.exp(-d / 3.0) * f, f.expand_as(d)], -1)
        return torch.cat([tok, extra], -1)

    def _view(self, own, opp, pres_own, pres_opp, glob, ball, ball_pres):
        B, L = own.shape[:2]
        if ball is None:
            ball = own.new_zeros(B, L, self.n_ball)
            ball_pres = torch.zeros(B, L, dtype=torch.bool, device=own.device)
        f = ball_pres[..., None].to(own.dtype)
        own, opp = self._with_ball(own, ball, ball_pres), self._with_ball(opp, ball, ball_pres)
        counts = torch.stack([pres_own.sum(-1), pres_opp.sum(-1)], -1).to(own.dtype) / 11.0
        z = torch.cat([self.own(own, pres_own), self.opp(opp, pres_opp), counts, glob, ball * f, f], -1)
        h, _ = self.gru(self.frame(z))
        return h

    def forward(self, own_h, opp_h, own_a, opp_a, pres_h, pres_a, glob_h, glob_a,
                ball_h=None, ball_a=None, ball_pres=None, return_aux=False, style=None):
        h_h = self._view(own_h, opp_h, pres_h, pres_a, glob_h, ball_h, ball_pres)
        h_a = self._view(own_a, opp_a, pres_a, pres_h, glob_a, ball_a, ball_pres)
        both = (h_h + h_a) / 2
        team, dead, bound = self.poss(h_h) - self.poss(h_a), self.dead(both), self.bound(both)
        if style is None:
            team, dead, bound = team[..., 0], dead[..., 0], bound[..., 0]
        else:
            idx = style.view(-1, 1, 1).expand(-1, team.shape[1], 1)
            team, dead, bound = (t.gather(-1, idx).squeeze(-1) for t in (team, dead, bound))
        self.last_bound = bound
        if return_aux:
            return team, dead, self.aux(h_h), self.aux(h_a), None, None, self.last_bound
        return team, dead
