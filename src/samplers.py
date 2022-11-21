from . import models, utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm

import logging
from typing import List, Optional
from IPython.core.debugger import Pdb

logger = logging.getLogger(__name__)


class SamplerList(nn.Module):
    def __init__(
        self,
        samplers: List[nn.Module],
    ):
        super().__init__()

        self.samplers = nn.ModuleList(samplers)

        self.hparams = {
            "samplers": utils.hparams(samplers),
        }

    @torch.no_grad()
    def forward(self, a, b, c, y):
        return torch.cat([sampler(a, b, c, y) for sampler in self.samplers], dim=-2)


class RandIntNbrWrapper(nn.Module):
    def __init__(
        self,
        sampler: nn.Module,
        randomize: bool = True,
        num_repeats: int = 1,
    ):
        super().__init__()

        self.sampler = sampler
        self.randomize = randomize
        self.num_repeats = num_repeats

        self.hparams = {
            "sampler": utils.hparams(sampler),
            "randomize": utils.hparams(randomize),
            "num_repeats": utils.hparams(num_repeats),
        }

    @torch.no_grad()
    def forward(self, a, b, c, y):
        # TODO: optimize
        raw = self.sampler(a, b, c, y)
        floor = torch.floor(raw)
        frac = raw - floor
        if self.randomize:
            neg_list = []
            for _ in range(self.num_repeats):
                rand = torch.rand_like(frac) <= frac
                neg_list.append(floor + rand)
            neg = torch.cat(neg_list, dim=-2)
        else:
            assert self.num_repeats == 1
            rand = 0.5 <= frac
            neg = floor + rand

        raw = torch.flatten(
            raw[:, None, :, :].expand(-1, self.num_repeats, -1, -1), 1, 2
        )
        diff = torch.abs(raw - neg)
        mx = torch.max(diff, dim=-1)[0][:, :, None]
        msk = torch.all(neg == y[:, None, :], dim=-1)[:, :, None]
        neg += (torch.sign(raw - neg) + (raw == neg)) * msk * (diff == mx)
        assert torch.all(torch.any(neg != y[:, None, :], dim=-1))
        return neg


class SolverSampler(nn.Module):
    def __init__(
        self,
        solver: nn.Module,
    ):
        super().__init__()
        self.solver = solver

        self.hparams = {
            "solver": utils.hparams(solver),
        }

    @torch.no_grad()
    def forward(self, a, b, c, y):
        a_l, a_k = a
        b_l, b_k = b
        yhat, status = self.solver(
            torch.cat([a_l, a_k], dim=-2), torch.cat([b_l, b_k], dim=-1), c, y,
        )
        return yhat[:, None, :]


class BatchSampler(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.hparams = {}

    @torch.no_grad()
    def forward(self, a, b, c, y):
        n = y.shape[0]
        idx = torch.remainder(
            torch.arange(n - 1)[None, :] + torch.arange(1, n + 1)[:, None], n
        )
        return y[idx]


class ProjSampler(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.hparams = {}

    @torch.no_grad()
    def forward(self, a, b, c, y):
        a_l, a_k = a
        b_l, b_k = b
        a = a_l
        b = b_l

        norm = torch.linalg.vector_norm(a, dim=-1)
        a = a / norm[:, :, None]
        b = b / norm
        a_t = a.transpose(-1, -2)
        b_t = b[:, None, :]

        dist = y[:, None, :] @ a_t + b_t
        proj = (y[:, :, None] - dist * a_t).transpose(-1, -2)
        return proj


class AllZerosSampler(nn.Module):
    def __init__(
            self,
            num_samples: int,
    ):
        super().__init__()
        self.num_samples = num_samples
            
    @torch.no_grad()
    def forward(self, a, b, c, y):
        return torch.zeros_like(y).unsqueeze(1).repeat(1,self.num_samples,1)

class AllOnesSampler(nn.Module):
    def __init__(
            self,
            num_samples: int,
    ):
        super().__init__()
        self.num_samples = num_samples

    @torch.no_grad()
    def forward(self, a, b, c, y):
        return torch.ones_like(y).unsqueeze(1).repeat(1,self.num_samples,1)





class UnitHypercubeSampler(nn.Module):
    def __init__(
        self,
        num_vars: int,
    ):
        super().__init__()

        l = torch.arange(1, 2 ** num_vars)
        r = torch.arange(num_vars)
        enum = (l[:, None] & (2 ** r)) > 0
        self.register_buffer("enum", enum.float())

    @torch.no_grad()
    def forward(self, a, b, c, y):
        return self.enum[None, :, :].expand(y.shape[0], -1, -1)


# XXX: deprecate
class NbrSampler(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    @torch.no_grad()
    def forward(self, a, b, c, y):
        y = y[:, None, :]
        e = torch.eye(y.shape[-1], device=y.device)
        return torch.cat([y + e, y - e], dim=-2)


class AltNbrSampler(nn.Module):
    def __init__(
        self,
        num_vars: int,
    ):
        super().__init__()

        self.register_buffer("eye", torch.eye(num_vars))

    @torch.no_grad()
    def forward(self, a, b, c, y):
        a_l, a_k = a
        b_l, b_k = b
        y = y[..., None, :]
        y_test = y + self.eye
        fdist_y_test = torch.sum(a_k * y_test, dim=-1) + b_k
        msk = torch.all(fdist_y_test >= 0, dim=-1)
        sgn = 2 * msk - 1
        return y + sgn * self.eye


class BitNbrSampler(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    @torch.no_grad()
    def forward(self, a, b, c, y):
        y = y[..., None, :]
        return y + (1 - 2 * y) * torch.eye(y.shape[-1], device=y.device)


class BitKHopSampler(nn.Module):
    def __init__(
        self,
        num_hops: int,
        num_samples: int,
    ):
        super().__init__()

        self.num_hops = num_hops
        self.num_samples = num_samples

        self.hparams = {
            "num_hops": utils.hparams(num_hops),
            "num_samples": utils.hparams(num_samples),
        }

    @torch.no_grad()
    def forward(self, a, b, c, y):
        batch_size, num_vars = y.shape
        idx = torch.randint(
            num_vars, (batch_size, self.num_samples, self.num_hops), device=y.device
        )
        mag = torch.any(F.one_hot(idx, num_classes=num_vars), dim=-2)
        y = y[..., None, :]
        return y + (1 - 2 * y) * mag


class KHopSampler(nn.Module):
    def __init__(
        self,
        num_hops: int,
        num_samples: int,
        one_hop_per_dim: bool = False,
    ):
        super().__init__()

        self.num_hops = num_hops
        self.num_samples = num_samples
        self.one_hop_per_dim = one_hop_per_dim

        self.hparams = {
            "num_hops": utils.hparams(num_hops),
            "num_samples": utils.hparams(num_samples),
            "one_hop_per_dim": utils.hparams(one_hop_per_dim),
        }

    @torch.no_grad()
    def forward(self, a, b, c, y):
        batch_size, num_vars = y.shape
        if self.one_hop_per_dim:
            idx = torch.topk(
                torch.rand(batch_size, self.num_samples, num_vars),
                self.num_hops,
                dim=-1,
            ).indices
        else:
            idx = torch.randint(
                num_vars, (batch_size, self.num_samples, self.num_hops), device=y.device
            )
        mag = torch.sum(F.one_hot(idx, num_classes=num_vars), dim=-2)
        msk = torch.randint(
            2, (batch_size, self.num_samples, num_vars), device=y.device
        )
        sgn = 2 * msk - 1
        return y[:, None, :] + sgn * mag
