"""Fit the tactics GRU: the base fit and the short adaptation fit that follows it.

Both take prepared feature dictionaries and return weights; neither reads or writes files.

Prepared data
-------------
``matches`` is a list with one entry per training copy of a match, each the output of
``prepare`` (``gru_features.match_features`` with the ball and the tactics features, then
``compact``). ``sources`` is a list of the same length with one ``(has_dead, label_style)``
pair per entry:

    has_dead      bool   the match's labels include dead ball, so its rows train the dead-ball head
    label_style   int    0 for the ranked label convention (DFL), 1 for any other source

Making the copies is the caller's job and is not reproduced here. The base fit used each
training match once as tracked and once degraded (seed 1000), with the labels of the
``controlled_v2_touch`` view. The adaptation fit used the four DFL training matches of the
fold only, each degraded twice (seeds 610 and 611, the ball left as tracked by the degrader)
and then with the ball hidden in gaps sampled from the A-League gap profile; ``clean`` in
``prepare`` is the untouched match, which supplies the true ball for the auxiliary target.
See ``inputs.py`` for the gap profile and the degraded copies of the frame models.

The sampler adds a ``spell_w`` array to every period and ``compact`` drops the opponent
arrays, so the feature dictionaries are changed in place.

Row rate
--------
Lengths in the settings are in 5 fps rows. ``STEP`` is the number of 25 fps frames between
rows (5 for the base fit, 2 for the adaptation fit); both fits set it, and ``prepare`` must
be called with the same step.
"""

import numpy as np
import torch

from possession.models import gru_features, team_shape
from possession.models.gru import TacticNet

KEYS = ("own_h", "opp_h", "own_a", "opp_a")
BALL_KEYS = ("ball_h", "ball_a", "tgt_h", "tgt_a")
STEP = 5

# The settings the base fit ran with (per held-out match; seed 0).
BASE = {
    "steps": 3000, "batch": 32, "length": 150, "lr": 2e-3, "wd": 1e-4,
    "width": 64, "hidden": 96, "layers": 2, "dropout": 0.1, "styles": True,
    "drop_slot": 0.05, "drop_frame": 0.03, "no_flip": False,
    "ball": True, "ball_off": 0.25, "touch": True, "thaw": True, "vel_span": 4, "step": 5,
    "spell_alpha": 0.7, "dfl_share": 0.5, "ema": 0.0,
    "aux": 0.2, "ptr": 0.0, "bound": 0.2, "smooth": 0.0, "seed": 0,
}
ADAPTATION = {"steps": 420, "batch": 8, "sample_step": 2, "seed": 20261113,
              "lr": 2e-4, "wd": 1e-4, "final_lr": 2e-5, "aux": 0.2, "bound": 0.2}


def rate():
    """Rows per default (5 fps) row: lengths given in 5 fps rows are multiplied by this."""
    return 5.0 / STEP


def probabilities(team_logit, dead_logit):
    """(..., 3) float: P(H), P(A), P(D)."""
    p_dead = torch.sigmoid(dead_logit)
    p_home = torch.sigmoid(team_logit)
    return torch.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], -1)


def spell_weights(label, alpha, ref_rows=28, lo=0.5, hi=4.0):
    """(L,) float: each H / A frame weighted by (ref / its possession's length) ** alpha, so short
    possessions count for more than their frame share. Spell F1 counts possessions, not frames.
    ref_rows is about the median possession (5.6 s at 5 fps)."""
    w = np.ones(len(label), dtype=np.float32)
    ref_rows = ref_rows * rate()
    if alpha <= 0:
        return w
    cut = np.flatnonzero(np.diff(label.astype(np.int64)) != 0) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(label)]):
        if label[a] <= gru_features.A:
            w[a:b] = np.clip((ref_rows / (b - a)) ** alpha, lo, hi)
    return w


def boundary_target(label, sigma=2.0):
    """(L,) float in [0, 1]: Gaussian bumps (sigma in rows) at every change of H/A/D state."""
    sigma = sigma * rate()
    lab = label.astype(np.int64)
    known = lab != gru_features.X
    change = np.zeros(len(lab), dtype=np.float32)
    idx = np.flatnonzero(known)
    if len(idx) > 1:
        flips = idx[1:][lab[idx[1:]] != lab[idx[:-1]]]
        change[flips] = 1.0
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    return np.clip(np.convolve(change, k, mode="same"), 0, 1)


def compact(f):
    """Shrink a match's features in place to about a quarter: drop the opponent arrays (they are
    the other team's own array turned half a turn) and store the rest as float16."""
    for p in f["periods"]:
        p.pop("opp_h", None)
        p.pop("opp_a", None)
        for k in ("own_h", "own_a", "glob_h", "glob_a"):
            if k in p:
                p[k] = p[k].astype(np.float16)
    return f


def turned(own):
    """A team's own tokens as its opponents see them: position, offset and velocity change sign."""
    out = own.astype(np.float32)
    out[..., :6] *= -1
    return out


def ball_gap_mask(rng, n, fully_off=0.25, cover=0.45, mean_rows=15):
    """(n,) bool, True where the ball is kept. A share of chunks has no ball at all (so the model
    stays a ball-free model); the rest lose about `cover` of rows in lognormal gaps."""
    mean_rows = mean_rows * rate()
    keep = np.ones(n, dtype=bool)
    if rng.random() < fully_off:
        keep[:] = False
        return keep
    target = rng.uniform(0, 2 * cover)
    while (~keep).mean() < target:
        g = max(1, int(rng.lognormal(np.log(mean_rows), 0.9)))
        s = rng.integers(0, n)
        keep[s:s + g] = False
    return keep


class Sampler:
    """Random fixed-length chunks from every period of every training match."""

    def __init__(self, feats, sources, length, rng, spell_alpha=0.0, dfl_share=0.0):
        self.items, self.weights = [], []
        style = []
        self.spell_alpha = spell_alpha
        for f, (has_dead, label_style) in zip(feats, sources, strict=True):
            for p in f["periods"]:
                n = len(p["label"])
                if n < length:
                    continue
                mask = gru_features.trainable(p["label"], p["pres_h"], p["pres_a"])
                p["spell_w"] = spell_weights(p["label"], spell_alpha)
                self.items.append((p, mask, has_dead))
                style.append(label_style)
                self.weights.append(n - length + 1)
        self.weights = np.array(self.weights, dtype=np.float64)
        self.style = np.array(style)
        ranked = self.style == 0
        if dfl_share > 0 and ranked.any() and (~ranked).any():
            # Draw a style-0 chunk with probability dfl_share, whatever its share of the frames.
            self.weights[ranked] *= dfl_share / self.weights[ranked].sum()
            self.weights[~ranked] *= (1 - dfl_share) / self.weights[~ranked].sum()
        self.weights /= self.weights.sum()
        self.length, self.rng = length, rng

    def batch(self, size, drop_slot, drop_frame, flip_y, with_ball=False, ball_off=0.25, tac=False):
        L = self.length
        picks = self.rng.choice(len(self.items), size=size, p=self.weights)
        out = {k: [] for k in KEYS + ("pres_h", "pres_a", "label", "mask", "dead_ok")}
        if with_ball:
            out.update({k: [] for k in BALL_KEYS + ("ball_pres", "tgt_pres", "near_h", "near_a")})
        out["bound"] = []
        out["spell_w"] = []
        out["style"] = self.style[picks]
        if tac:
            out.update({"glob_h": [], "glob_a": []})
        for i in picks:
            p, mask, has_dead = self.items[i]
            s = self.rng.integers(0, len(p["label"]) - L + 1)
            sl = slice(s, s + L)
            pres_h, pres_a = p["pres_h"][sl].copy(), p["pres_a"][sl].copy()
            # Player dropout: whole slots for the chunk, plus scattered frames. The competition
            # tracks about 20.5 of 22 players on a graded frame.
            for pres in (pres_h, pres_a):
                pres &= self.rng.random(pres.shape[1]) >= drop_slot
                pres &= self.rng.random(pres.shape) >= drop_frame
            oh, oa = p["own_h"][sl].astype(np.float32), p["own_a"][sl].astype(np.float32)
            sets = [oh, turned(oa), oa, turned(oh)]
            balls = [p[k][sl].copy() for k in BALL_KEYS] if with_ball else []
            globs = [p[k][sl].astype(np.float32) for k in ("glob_h", "glob_a")] if tac else []
            if flip_y and self.rng.random() < 0.5:
                for s_ in sets:
                    s_[..., [1, 3, 5]] *= -1
                for b_ in balls:
                    b_[..., [1, 3]] *= -1
                for g_ in globs:
                    g_[..., list(team_shape.Y_SIGNED)] *= -1
            for k, v in zip(("glob_h", "glob_a"), globs):
                out[k].append(v)
            for k, v in zip(KEYS, sets):
                out[k].append(v)
            if with_ball:
                bp = p["ball_pres"][sl] & ball_gap_mask(self.rng, L, fully_off=ball_off)
                balls[0][~bp] = 0.0
                balls[1][~bp] = 0.0
                for k, v in zip(BALL_KEYS, balls):
                    out[k].append(v)
                out["ball_pres"].append(bp)
                out["tgt_pres"].append(p["tgt_pres"][sl].copy())
                both = np.concatenate([pres_h, pres_a], 1)
                for k, order in (("near_h", both), ("near_a", np.concatenate([pres_a, pres_h], 1))):
                    nr = p[k][sl].copy()
                    ok = nr >= 0
                    ok[ok] = order[np.flatnonzero(ok), nr[ok]]
                    nr[~ok] = -1
                    out[k].append(nr)
            out["bound"].append(boundary_target(p["label"][sl]))
            out["spell_w"].append(p["spell_w"][sl])
            out["pres_h"].append(pres_h)
            out["pres_a"].append(pres_a)
            out["label"].append(p["label"][sl])
            out["mask"].append(mask[sl] & pres_h.any(-1) & pres_a.any(-1))
            out["dead_ok"].append(np.full(L, has_dead))
        return {k: (v if isinstance(v, np.ndarray) else np.stack(v)) for k, v in out.items()}


def to_torch(batch, device):
    t = {}
    for k, v in batch.items():
        if v.dtype == np.float32:
            t[k] = torch.from_numpy(v).to(device)
        elif v.dtype == np.bool_:
            t[k] = torch.from_numpy(v).to(device)
        else:
            t[k] = torch.from_numpy(v.astype(np.int64)).to(device)
    return t


def loss_fn(team, dead, b, aux=None, aux_w=0.0):
    y = b["label"]
    m = b["mask"]
    is_dead = (y == gru_features.D).float()
    dmask = m & b["dead_ok"]
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    l_dead = (bce(dead, is_dead, reduction="none") * dmask).sum() / dmask.sum().clamp(min=1)
    alive = m & (y <= gru_features.A)
    tw = alive * b["spell_w"]
    l_team = (bce(team, (y == gru_features.H).float(), reduction="none") * tw).sum() / tw.sum().clamp(min=1)
    total = l_dead + l_team
    w = aux_w if isinstance(aux_w, dict) else {"aux": aux_w}
    if aux is not None and w.get("aux", 0) > 0:
        tm = (b["tgt_pres"] & m).unsqueeze(-1).float()
        err = sum(((a - b[k][..., :2]) ** 2 * tm).sum() for a, k in zip(aux[:2], ("tgt_h", "tgt_a")))
        total = total + w["aux"] * err / (2 * tm.sum().clamp(min=1))
    if aux is not None and len(aux) > 2:
        ptr_h, ptr_a, bound = aux[2:]
        if w.get("ptr", 0) > 0 and ptr_h is not None:
            ce = 0.0
            for logits, k, pres in ((ptr_h, "near_h", torch.cat([b["pres_h"], b["pres_a"]], -1)),
                                    (ptr_a, "near_a", torch.cat([b["pres_a"], b["pres_h"]], -1))):
                tgt = b[k]
                ok = tgt >= 0
                if ok.any():
                    lg = logits.masked_fill(~pres, -1e4)[ok]
                    ce = ce + torch.nn.functional.cross_entropy(lg, tgt[ok])
            total = total + w["ptr"] * ce / 2
        if w.get("bound", 0) > 0:
            bm = m.float()
            lb = (bce(bound, b["bound"], reduction="none") * bm).sum() / bm.sum().clamp(min=1)
            total = total + w["bound"] * lb
    if w.get("smooth", 0) > 0:
        logp = torch.log(probabilities(team, dead).clamp(min=1e-6))
        d = (logp[:, 1:] - logp[:, :-1]).clamp(-4, 4) ** 2
        mm = (m[:, 1:] & m[:, :-1]).unsqueeze(-1).float()
        total = total + w["smooth"] * (d * mm).sum() / (3 * mm.sum().clamp(min=1))
    return total, l_dead.item(), l_team.item()


def run_net(net, b, with_ball, return_aux=False):
    args = [b[k] for k in KEYS] + [b["pres_h"], b["pres_a"], b["glob_h"], b["glob_a"]]
    style = b.get("style") if getattr(net, "styles", 1) > 1 else None
    if with_ball:
        return net(*args, b["ball_h"], b["ball_a"], b["ball_pres"], return_aux=return_aux, style=style)
    return net(*args, return_aux=return_aux, style=style)


def build(settings=BASE, device="cpu"):
    """The network the settings describe, freshly initialised."""
    return TacticNet(
        settings["width"], settings["hidden"], settings["layers"], settings["dropout"],
        styles=2 if settings["styles"] else 1,
        n_ball=4 + (gru_features.N_TOUCH if settings["touch"] else 0),
    ).to(device)


def prepare(match, settings=BASE, step=None, clean=None):
    """One training copy of a harmonised match as the sampler expects it.

    ``match`` is the (possibly degraded) match the model sees; ``clean`` is the same match
    untouched, for the true-ball auxiliary target (omit it when ``match`` is untouched).
    ``step`` defaults to the base fit's; pass ``ADAPTATION["sample_step"]`` for the adaptation.
    """
    step = settings["step"] if step is None else step
    return compact(gru_features.match_features(
        match, step=step, vel_span=settings["vel_span"], with_ball=True, target=clean, tactics=True,
        thaw=settings["thaw"], touch=settings["touch"], delta_rows=round(25 / step),
    ))


def fit_base(matches, sources, settings=BASE, device="cpu"):
    """Initialise a network and fit it on the prepared ``matches``.

    Returns ``(net, last)``. ``net`` holds the weights to keep: the running average when
    ``settings["ema"]`` is positive, else the final weights. ``last`` is the final state
    dictionary before the average was loaded, or None without averaging.
    """
    global STEP
    STEP = settings["step"]
    device = torch.device(device)
    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    net = build(settings, device)
    with_ball = settings["ball"] or settings["aux"] > 0
    sampler = Sampler(matches, sources, int(settings["length"] * rate()), rng,
                      settings["spell_alpha"], settings["dfl_share"])
    opt = torch.optim.AdamW(net.parameters(), lr=settings["lr"], weight_decay=settings["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=settings["lr"], total_steps=settings["steps"],
                                                pct_start=0.1)
    ema = None
    if settings["ema"] > 0:
        ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    weights = {key: settings[key] for key in ("aux", "ptr", "bound", "smooth")}
    for step in range(1, settings["steps"] + 1):
        net.train()
        b = to_torch(sampler.batch(settings["batch"], settings["drop_slot"], settings["drop_frame"],
                                   not settings["no_flip"], with_ball, settings["ball_off"], True), device)
        if not settings["ball"] and with_ball:
            b["ball_pres"] = torch.zeros_like(b["ball_pres"])
        outs = run_net(net, b, with_ball, return_aux=True)
        loss, _, _ = loss_fn(outs[0], outs[1], b, outs[2:], weights)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            decay = min(settings["ema"], (1 + step) / (10 + step))
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(decay).add_(v.detach(), alpha=1 - decay)
                    else:
                        ema[k].copy_(v)
    last = None
    if ema is not None:
        last = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(ema)
    return net, last


def fit_adaptation(model, matches, sources, settings=BASE, adaptation=ADAPTATION, device="cpu"):
    """Continue fitting ``model`` (already holding the base weights) on the prepared ``matches``.

    ``settings`` are the base fit's: its chunk length, spell weighting and player dropout are
    reused. The ball is always given where the copy still has it (no ball-free chunks), chunks
    are mirrored across the pitch at random, and the loss is the possession and dead-ball
    terms plus the true-ball and boundary terms. The held-out match must not be in ``matches``.
    ``model`` is fitted in place and returned.
    """
    global STEP
    STEP = adaptation["sample_step"]
    device = torch.device(device)
    torch.manual_seed(adaptation["seed"])
    sampler = Sampler(matches, sources, int(settings["length"] * rate()),
                      np.random.default_rng(adaptation["seed"]), settings["spell_alpha"])
    opt = torch.optim.AdamW(model.parameters(), lr=adaptation["lr"], weight_decay=adaptation["wd"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, adaptation["steps"], eta_min=adaptation["final_lr"])
    weights = {"aux": adaptation["aux"], "bound": adaptation["bound"]}
    model.train()
    for step in range(1, adaptation["steps"] + 1):
        b = to_torch(sampler.batch(adaptation["batch"], settings["drop_slot"], settings["drop_frame"], True,
                                   with_ball=True, ball_off=0., tac=True), device)
        out = run_net(model, b, True, return_aux=True)
        loss, _, _ = loss_fn(out[0], out[1], b, out[2:], weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        schedule.step()
    return model
