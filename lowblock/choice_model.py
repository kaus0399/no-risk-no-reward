"""Graph attention choice model.

For each decision moment, the model reads the positions and velocities of the 22 players and the ball as a fully
connected graph and predicts which option the attacking team takes in the next five seconds (recycle, pass or carry
into the block, cross). Its predicted probabilities are the propensities used by the doubly robust estimator in
lowblock/aipw.py.

Node order: 11 attackers (goalkeeper first), 11 defenders (goalkeeper first), then the ball. Coordinates are metres
in the attacking direction: depth from the goal line being attacked and lateral position from the centre line. Missing players
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


def validation_matches(match_ids, season, train_season, seed=7071):
    """The 15% of the training season's matches held back for early stopping."""
    import numpy as np

    um = np.sort(np.unique(match_ids[season == train_season]))
    return set(np.random.default_rng(seed).choice(um, max(1, int(round(0.15 * len(um)))), replace=False).tolist())


def train_season_swap(W, M, ctx, A, season, match_ids, train_season, seed=7071, epochs=25, minutes=12.0,
                      batch_size=256, device="cpu", log=print):
    """Train on `train_season` (15% of its matches held back for early stopping) and return probabilities [n, 3] for
    the rows of the other season (NaN elsewhere) plus a training summary."""
    import time

    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    itr_all = np.where(season == train_season)[0]
    ite = np.where(season != train_season)[0]
    va_m = validation_matches(match_ids, season, train_season, seed)
    is_va = np.isin(match_ids, list(va_m))
    itr, iva = itr_all[~is_va[itr_all]], itr_all[is_va[itr_all]]
    mu, sd = np.nanmean(ctx[itr], 0), np.nanstd(ctx[itr], 0) + 1e-6
    c = np.nan_to_num((ctx - mu) / sd, nan=0.0).astype(np.float32)
    dev = torch.device(device)
    Wd, Md = torch.from_numpy(W).to(dev), torch.from_numpy(M).to(dev)
    Cd, Yd = torch.from_numpy(c).to(dev), torch.from_numpy(A.astype(np.int64)).to(dev)
    net = ChoiceModel().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)

    def batches(idx, shuffle):
        o = np.random.permutation(idx) if shuffle else idx
        for i in range(0, len(o), batch_size):
            yield torch.from_numpy(np.sort(o[i:i + batch_size])).to(dev)

    def predict(idx):
        net.eval()
        out = []
        with torch.no_grad():
            for j in batches(idx, False):
                out.append(torch.softmax(net(Wd[j].float(), Md[j], Cd[j]), -1).float().cpu().numpy())
        return np.concatenate(out)

    best, bad, best_state, t0, hist = np.inf, 0, None, time.time(), []
    for ep in range(epochs):
        te = time.time()
        net.train()
        tot, n = 0.0, 0
        for j in batches(itr, True):
            loss = F.cross_entropy(net(Wd[j].float(), Md[j], Cd[j]), Yd[j])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(j)
            n += len(j)
        Pv = predict(iva)
        lv = float(-np.mean(np.log(np.clip(Pv[np.arange(len(iva)), A[iva]], 1e-12, 1))))
        hist.append({"epoch": ep + 1, "train_logloss": tot / max(n, 1), "val_logloss": lv})
        log(f"[train {train_season}] epoch {ep + 1}: train log loss {tot / max(n, 1):.4f} | val log loss {lv:.4f}")
        if lv < best - 1e-4:
            best, bad = lv, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if bad >= 3 or (time.time() - t0) / 60 + (time.time() - te) / 60 > minutes:
            break
    net.load_state_dict(best_state)
    P = np.full((len(A), 3), np.nan)
    P[ite] = predict(ite)
    return P, {"train_season": train_season, "n_train": int(len(itr)), "n_val": int(len(iva)), "n_test": int(len(ite)),
               "epochs": hist, "best_val_logloss": best, "val_matches": sorted(int(x) for x in va_m)}
