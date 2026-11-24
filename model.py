import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp
from config import SIZES

# H3 layout: v_body, mD fraction, flash count {0,1,2+}, tip-speed bin {4}, log rotor rate
H3_V, H3_MD, H3_FLASH, H3_TIP, H3_RATE = 0, 1, slice(2, 5), slice(5, 9), 9


class Net(nn.Module):
    """Conv-only CNN on the 1x150 Doppler spectrum (dim=1) or the repo's 5x150 map (dim=2).
    FiLM: each block's BN output is scaled/shifted by gamma, beta = MLP(waveform vector)."""

    def __init__(self, size="S", dim=1, film=True):
        super().__init__()
        self.dim, self.film = dim, film
        c = SIZES[size]
        conv, bn, pool = (nn.Conv1d, nn.BatchNorm1d, nn.MaxPool1d(2)) if dim == 1 else (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d((1, 2)))
        k, pad = (5, 2) if dim == 1 else ((3, 5), (1, 2))
        self.convs = nn.ModuleList(conv(i, o, k, padding=pad) for i, o in zip((1,) + c[:2], c))
        self.bns = nn.ModuleList(bn(o) for o in c)
        self.films = nn.ModuleList(nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2 * o)) for o in c)
        for f in self.films:
            nn.init.zeros_(f[2].weight)
            nn.init.zeros_(f[2].bias)
        self.pool = pool
        self.drop = nn.Dropout(0.2)
        self.h1 = nn.Linear(c[2], 3)
        self.h2 = nn.Linear(c[2], 6)
        self.h3 = nn.Linear(c[2], 10)

    def forward(self, x, w):
        x = x.unsqueeze(1)
        for i in range(3):
            x = self.bns[i](self.convs[i](x))
            if self.film:
                g, b = self.films[i](w).chunk(2, 1)
                shape = g.shape + (1,) * (x.dim() - 2)
                x = x * (1 + g.view(shape)) + b.view(shape)
            x = torch.relu(x)
            if i < 2:
                x = self.pool(x)
        z = x.flatten(2).mean(2)
        zd = self.drop(z)
        return dict(z=z, h1=self.h1(zd), h2=self.h2(zd), h3=self.h3(zd))


def baseline():
    """The repo's original model (5x150 input, binary output)."""
    return nn.Sequential(
        nn.Conv2d(1, 16, (3, 5), padding=(1, 2)), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((1, 2)),
        nn.Conv2d(16, 32, (3, 5), padding=(1, 2)), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1, 2)),
        nn.Conv2d(32, 64, (3, 5), padding=(1, 2)), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2),
    )


def fit_openset(z_tr, t_tr, z_va, h2_va, known):
    """Prototypes per known drone type with a tied covariance (Mahalanobis) plus the energy of the H2
    logits over known types. Both are z-normalized on validation drones; tau = their 95th percentile."""
    means = np.stack([z_tr[t_tr == k].mean(0) for k in known])
    cen = np.concatenate([z_tr[t_tr == k] - means[i] for i, k in enumerate(known)])
    os = dict(known=list(known), means=means, prec=np.linalg.pinv(np.cov(cen.T)))
    e, m = raw_scores(os, z_va, h2_va)
    os.update(e_mu=e.mean(), e_sd=e.std(), m_mu=m.mean(), m_sd=m.std())
    os["tau"] = np.percentile(score(os, z_va, h2_va), 95)
    return os


def raw_scores(os, z, h2):
    energy = -logsumexp(h2[:, os["known"]], axis=1)
    diff = z[:, None] - os["means"][None]
    maha = np.einsum("nkd,de,nke->nk", diff, os["prec"], diff).min(1)
    return energy, maha


def score(os, z, h2):
    e, m = raw_scores(os, z, h2)
    return (e - os["e_mu"]) / os["e_sd"] + (m - os["m_mu"]) / os["m_sd"]
