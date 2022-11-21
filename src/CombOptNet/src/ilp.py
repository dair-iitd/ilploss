from . import utils

import gurobi as grb
from gurobi import GRB
import numpy as np
import torch
from torch import autograd, nn
import torch.nn.functional as F

from tqdm.auto import tqdm

import collections
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

STATUS_MSG = collections.defaultdict(
    lambda: "unknown",
    {
        1: "loaded",
        2: "optimal",
        3: "infeasible",
        4: "inf_or_unbd",
        5: "unbounded",
        6: "cutoff",
        7: "iteration_limit",
        8: "node_limit",
        9: "time_limit",
        10: "solution_limit",
        11: "interrupted",
        12: "numeric",
        13: "suboptimal",
        14: "inprogress",
        15: "user_obj_limit",
    },
)


class DiffSolve(autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, obj, *args):
        *args, h = args
        args = [arg.detach() for arg in args]
        if h is None:
            y, status = obj.solve(*args)
        else:
            y, status = obj.solve(*args, h)
        ctx.save_for_backward(*args, y.detach(), status.detach())
        ctx.obj = obj
        return y, status

    @staticmethod
    @torch.no_grad()
    def make_basis(y, dy):
        device = y.device
        batch_size, num_vars = y.shape
        val, idx = torch.sort(torch.abs(dy), dim=-1, descending=True)

        tril = torch.tril(torch.ones(num_vars, num_vars, device=device)).expand(
            batch_size, -1, -1
        )
        basis = torch.empty(batch_size, num_vars, num_vars, device=device)
        basis.scatter_(-1, idx[:, None, :].expand(-1, num_vars, -1), tril)
        basis *= torch.sign(dy)[:, None, :]

        pad = torch.cat([val, torch.zeros(batch_size, 1, device=device)], dim=-1)
        coeff = pad[:, :-1] - pad[:, 1:]

        return basis, coeff

    @staticmethod
    @torch.enable_grad()
    def backward(ctx, dy, dstatus):
        a, b, c, y, status = ctx.saved_tensors
        num_constrs, num_vars = a.shape[-2:]

        basis, coeff = DiffSolve.make_basis(y, dy)
        pos = y[:, None, :] - basis

        neg = y[:, None, None, :].expand(-1, num_vars, -1, -1)
        a = a[:, None, :, :].expand(-1, num_vars, -1, -1)
        b = b[:, None, :].expand(-1, num_vars, -1)
        c = c[:, None, :].expand(-1, num_vars, -1)

        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True

        loss = ctx.obj.criterion(a, b, c, pos, neg)
        loss.backward()

        da = torch.sum(coeff[:, :, None, None] * a.grad, dim=1)
        db = torch.sum(coeff[:, :, None] * b.grad, dim=1)
        dc = torch.sum(coeff[:, :, None] * c.grad, dim=1)

        return None, da, db, dc, None


class ILPLoss(nn.Module):
    def __init__(
        self,
        tau: float = 1.0,
    ):
        super().__init__()

        self.tau = tau

        self.hparams = {
            "tau": tau,
        }

    def min(self, x, dim):
        return -self.tau * torch.logsumexp(-x / self.tau, dim=dim)

    def forward(self, a, b, c, pos, neg):
        """
        a: *batch x constr x var
        b: *batch x constr
        c: *batch x var
        pos: *batch x var
        neg: *batch x neg x var
        """

        norm_a = torch.linalg.vector_norm(a, dim=-1)
        a = a / norm_a[..., :, None]
        b = b / norm_a
        dist_pos = torch.sum(a * pos[..., None, :], dim=-1) + b
        dist_neg = neg @ a.transpose(-1, -2) + b[..., None, :]
        """
        norm_a: *batch x constr
        dist_pos: *batch x constr
        dist_neg: *batch x neg x constr
        """

        norm_c = torch.linalg.vector_norm(c, dim=-1)
        c = c / norm_c[..., None]
        dist_obj = torch.sum(c[..., None, :] * (pos[..., None, :] - neg), dim=-1)
        """
        norm_c: *batch
        dist_obj: *batch x neg
        """

        loss_pos = torch.sum(F.relu(-dist_pos), dim=-1)
        err_pos = torch.max(F.relu(-dist_pos), dim=-1).values
        margin_pos = torch.min(F.relu(dist_pos), dim=-1).values
        self.model().log_dict(
            {
                "pos/loss": torch.mean(loss_pos),
                "pos/err": torch.mean(err_pos),
                "pos/margin": torch.mean(margin_pos),
            },
        )

        loss_neg = torch.mean(self.min(F.relu(dist_neg), dim=-1), dim=-1)
        err_neg = torch.max(torch.min(F.relu(dist_neg), dim=-1).values, dim=-1).values
        margin_neg = torch.min(
            torch.max(F.relu(-dist_neg), dim=-1).values, dim=-1
        ).values
        self.model().log_dict(
            {
                "neg/loss": torch.mean(loss_neg),
                "neg/err": torch.mean(err_neg),
                "neg/margin": torch.mean(margin_neg),
            },
        )

        loss_obj = torch.mean(F.relu(dist_obj), dim=-1)
        err_obj = torch.max(F.relu(dist_obj), dim=-1).values
        margin_obj = torch.min(F.relu(-dist_obj), dim=-1).values
        self.model().log_dict(
            {
                "obj/loss": torch.mean(loss_obj),
                "obj/err": torch.mean(err_obj),
                "obj/margin": torch.mean(margin_obj),
            },
        )

        loss = loss_pos + (loss_neg + loss_obj) * (loss_pos == 0)
        self.model().log_dict(
            {
                "loss": torch.mean(loss),
            }
        )

        return torch.mean(loss)


class CombOptNet(nn.Module):
    def __init__(
        self,
        vtype: str = GRB.INTEGER,
        env: Optional[grb.Env] = None,
        num_workers: int = 1,
        show_tqdm: bool = False,
        criterion: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.vtype = vtype
        self.env = grb.Env() if env is None else env
        self.exe = ThreadPoolExecutor(num_workers)
        self.show_tqdm = show_tqdm
        if criterion is None:
            criterion = ILPLoss()
        self.criterion = criterion

        self.hparams = {
            "criterion": utils.hparams(criterion),
        }

    def solve(self, *args):
        a = args[0]
        # TODO: *batch_size
        batch_size, _, num_vars = a.shape
        y = torch.empty(batch_size, num_vars, device=a.device)
        status = torch.empty(batch_size, device=a.device, dtype=torch.long)

        def aux(i, a, b, c, h=None):
            num_constrs, num_vars = a.shape
            m = grb.Model(env=self.env)
            y_obj = m.addMVar(num_vars, float("-inf"), float("inf"), vtype=self.vtype)
            m.addMConstr(a, y_obj, GRB.GREATER_EQUAL, -b)
            m.setMObjective(None, c, 0.0, sense=GRB.MINIMIZE)
            if h is not None:
                y_obj.varHintVal = h
            m.optimize()
            try:
                y[i] = torch.as_tensor(y_obj.x)
            except grb.GurobiError:
                # y[i] = torch.nan
                # XXX
                y[i] = torch.zeros(num_vars)
            status[i] = m.status

        list(
            tqdm(
                self.exe.map(
                    aux,
                    np.arange(batch_size),
                    *[arg.cpu().numpy() for arg in args],
                ),
                "instances",
                total=batch_size,
                disable=not self.show_tqdm,
                leave=False,
            )
        )

        return y, status

    def forward(self, *args):
        return DiffSolve.apply(self, *args)
