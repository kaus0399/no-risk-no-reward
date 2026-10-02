"""Graph attention choice model.

For each decision moment, the model reads the positions and velocities of the 22 players and the ball as a fully
connected graph and predicts which option the attacking team takes in the next five seconds (recycle, pass or carry
into the block, cross). Its predicted probabilities are the propensities used by the doubly robust estimator in
lowblock/aipw.py.

Node order: 11 attackers (goalkeeper first), 11 defenders (goalkeeper first), then the ball. Coordinates are metres
in the attacking direction: depth from the attacked goal and lateral position from the centre line. Missing players
are masked.

Training: cross-entropy, trained on one season and evaluated on the other (and vice versa), so every prediction is
out of season.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IS_ATT = torch.tensor([1.0] * 11 + [0.0] * 12)
IS_DEF = torch.tensor([0.0] * 11 + [1.0] * 11 + [0.0])
IS_GK = torch.tensor([1.0] + [0.0] * 10 + [1.0] + [0.0] * 11)
IS_BALL = torch.tensor([0.0] * 22 + [1.0])


def node_edge(x, m):
    """x [G, 23, 4] (depth, lateral, vx, vy) and mask m [G, 23] -> node features [G, 23, 14], edge features [G, 23, 23, 2]."""
    dev = x.device
    pos = x[..., :2]
    rel = pos[:, 22:23] - pos                     # vector from each node to the ball
    db = rel.norm(dim=-1, keepdim=True)
    ub = torch.where(db > 1e-3, rel / db.clamp_min(1e-3), torch.zeros_like(rel))
    dg = pos.norm(dim=-1, keepdim=True)           # distance to the centre of the goal
    ug = -pos / dg.clamp_min(1e-3)
    G = x.shape[0]
    flags = torch.stack([IS_ATT, IS_DEF, IS_GK, IS_BALL], -1).to(dev)[None].expand(G, -1, -1)
    f = torch.cat([x[..., :1] / 52.5, x[..., 1:2] / 34.0, x[..., 2:4] / 8.0, flags, db / 40.0, ub, dg / 52.5, ug], -1)
    f = f * m[..., None]
    d = torch.cdist(pos, pos)
    e = torch.stack([d / 20.0, torch.exp(-d / 5.0)], -1)
    return f, e


class GATv2Layer(nn.Module):
    """Dense GATv2 attention (Brody et al., 2022) over a fully connected graph with edge features."""

    def __init__(self, d=64, heads=4, de=2, drop=0.1):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.wl, self.wr = nn.Linear(d, d), nn.Linear(d, d)
        self.we = nn.Linear(de, d, bias=False)
        self.att = nn.Parameter(torch.randn(heads, self.dh) * (self.dh ** -0.5))
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, h, e, mask):
        G, N, _ = h.shape
        xl = self.wl(h).view(G, N, self.h, self.dh)
        xr = self.wr(h).view(G, N, self.h, self.dh)
        s = F.leaky_relu(xl[:, :, None] + xr[:, None, :] + self.we(e).view(G, N, N, self.h, self.dh), 0.2)
        logits = (s * self.att).sum(-1).masked_fill(~mask[:, None, :, None], -1e4)
        alpha = self.drop(torch.softmax(logits, dim=2))
        o = torch.einsum("gijh,gjhd->gihd", alpha, xr).reshape(G, N, -1)
        return self.norm(h + self.drop(F.elu(self.out(o))))


class Encoder(nn.Module):
    """Three GATv2 layers, then masked mean and max pooling over nodes."""

    def __init__(self, d=64, layers=3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(14, d), nn.ELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList([GATv2Layer(d) for _ in range(layers)])

    def forward(self, x, m):
        f, e = node_edge(x, m)
        h = self.inp(f)
        for layer in self.layers:
            h = layer(h, e, m)
        mf = m[..., None].float()
        mean = (h * mf).sum(1) / mf.sum(1).clamp_min(1.0)
        mx = h.masked_fill(~m[..., None], -1e4).max(1).values
        mx = torch.where(m.any(1, keepdim=True), mx, torch.zeros_like(mx))
        return torch.cat([mean, mx], -1)


class ChoiceModel(nn.Module):
    """Graph encoding of the decision picture plus match context -> logits over the three options.

    Context (7 values): minute, score difference, attacking and defending team strength, whether the possession
    started from a regain, whether it started from a throw-in, and seconds since the block set.
    """

    def __init__(self, d=64, n_context=7, n_options=3):
        super().__init__()
        self.encoder = Encoder(d)
        self.head = nn.Sequential(nn.Linear(2 * d + n_context, d), nn.ELU(), nn.Dropout(0.1), nn.Linear(d, n_options))

    def forward(self, x, m, context):
        return self.head(torch.cat([self.encoder(x, m), context], -1))
