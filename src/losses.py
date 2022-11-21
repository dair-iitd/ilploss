from . import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm

import logging
from pathlib import Path
from typing import Callable, List, Optional

from .utils import compute_normalized_solution, compute_denormalized_solution

logger = logging.getLogger(__name__)


class NormalizedMSELoss(nn.Module):
    def __init__(
        self,
        loss_fn: nn.Module,
        lb=0,
        ub=1,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.lb = lb
        self.ub = ub

    def forward(self, yhat, y):
        yhat_norm = compute_normalized_solution(yhat, self.lb, self.ub)
        y_norm = compute_normalized_solution(y, self.lb, self.ub)
        return self.loss_fn(yhat_norm, y_norm)


class ILPLoss(nn.Module):
    def __init__(
        self,
        balancer: nn.Module,
        pos_param: float,
        neg_param: float,
        criterion: str = "hinge",
    ):
        super().__init__()
        self.balancer = balancer
        self.pos_param = pos_param
        self.neg_param = neg_param
        self.criterion = criterion

        self.hparams = {
            "balancer": utils.hparams(balancer),
            "pos_param": utils.hparams(pos_param),
            "neg_param": utils.hparams(neg_param),
            "criterion": utils.hparams(criterion),
        }

    def forward(self, a, b, c, pos, neg):
        a_l, a_k = a
        b_l, b_k = b
        """
        a_l, a_k: *batch x constr x var
        b_l, b_k: *batch x constr
        c: *batch x var
        pos: *batch x var
        neg: *batch x neg x var
        """

        known_c = not c.requires_grad and c.grad_fn is None

        if known_c:
            a_k = torch.cat([a_k, -c[..., None, :]], dim=-2)
            b_k = torch.cat([b_k, torch.sum(c * pos, dim=-1)[..., None]], dim=-1)
        else:
            a_l = torch.cat([a_l, -c[..., None, :]], dim=-2)
            b_l = torch.cat([b_l, torch.sum(c * pos, dim=-1)[..., None]], dim=-1)

        norm_l = torch.linalg.vector_norm(a_l, dim=-1)
        a_l = a_l / norm_l[..., :, None]
        b_l = b_l / norm_l
        dist_pos_l = torch.sum(a_l * pos[..., None, :], dim=-1) + b_l
        dist_neg_l = neg @ a_l.transpose(-1, -2) + b_l[..., None, :]
        if not known_c:
            dist_pos_l[..., -1] = 0
        """
        norm_l: *batch x constr
        dist_pos_l: *batch x constr
        dist_neg_l: *batch x neg x constr
        """

        fdist_pos_k = torch.sum(a_k * pos[..., None, :], dim=-1) + b_k
        fdist_neg_k = neg @ a_k.transpose(-1, -2) + b_k[..., None, :]
        """
        fdist_pos_k: *batch x constr
        fdist_neg_k: *batch x neg x constr
        """

        msk_pos = torch.any(neg != pos[..., None, :], dim=-1)
        msk_known = torch.all(fdist_neg_k >= 0, dim=-1)
        msk = msk_pos & msk_known

        do_log = self.model().global_step % self.model().trainer.log_every_n_steps == 0

        if do_log:
            self.model().log_dict(
                {
                    "msk": torch.mean(msk.float()),
                    "msk/pos": torch.mean(msk_pos.float()),
                    "msk/known": torch.mean(msk_known.float()),
                    "msk/use": torch.mean(
                        msk * torch.all(dist_neg_l >= 0, dim=-1).float()
                    ),
                }
            )

            acc_pos_l = torch.all(dist_pos_l >= 0, dim=-1)
            acc_pos_k = torch.all(fdist_pos_k >= 0, dim=-1)
            if not torch.all(acc_pos_k):
                logger.warning(f"+ve example violates known constraints!")
            acc_pos = acc_pos_l & acc_pos_k

            acc_neg_frac = torch.sum(
                torch.any(dist_neg_l < 0, dim=-1) * msk,
                dim=-1,
            ) / torch.sum(msk, dim=-1)
            acc_neg = acc_neg_frac == 1

            acc = acc_pos & acc_neg

            self.model().log_dict(
                {
                    "acc": torch.mean(acc.float()),
                    "acc/pos": torch.mean(acc_pos.float()),
                    "acc/neg": torch.mean(acc_neg.float()),
                    "acc/neg_frac": torch.mean(acc_neg_frac),
                },
            )

        if self.criterion == "hinge":
            if known_c:
                loss_pos = torch.mean(F.relu(self.pos_param - dist_pos_l), dim=-1)
            elif dist_pos_l.shape[-1] > 1:
                loss_pos = torch.mean(
                    F.relu(self.pos_param - dist_pos_l[..., :-1]), dim=-1
                )
            else:
                loss_pos = torch.tensor(0.0, device=dist_pos_l.device)

        elif self.criterion == "cross_entropy":
            if known_c:
                loss_pos = torch.mean(F.softplus(-dist_pos_l / self.pos_param), dim=-1)
            elif dist_pos_l.shape[-1] > 1:
                loss_pos = torch.mean(
                    F.softplus(-dist_pos_l[..., :-1] / self.pos_param), dim=-1
                )
            else:
                loss_pos = torch.tensor(0.0, device=dist_pos_l.device)

        if do_log:
            if known_c:
                min_dist_pos = torch.min(dist_pos_l, dim=-1).values
            elif dist_pos_l.shape[-1] > 1:
                min_dist_pos = torch.min(dist_pos_l[..., :-1], dim=-1).values
            else:
                min_dist_pos = torch.tensor(float("nan"))
            err_pos = F.relu(-min_dist_pos)
            margin_pos = F.relu(min_dist_pos)

            self.model().log_dict(
                {
                    "pos/loss": torch.mean(loss_pos),
                    "pos/err": torch.mean(err_pos),
                    "pos/margin": torch.mean(margin_pos),
                },
            )

        temp = self.model().query_scheduler("temp")

        if do_log:
            self.model().log("temp", temp)

        with torch.no_grad():
            if temp <= torch.finfo().eps:
                w = utils.one_hot_argmin(dist_neg_l, dim=-1)
            else:
                w = F.softmin(dist_neg_l / temp, dim=-1)

        if self.criterion == "hinge":
            loss_neg = torch.sum(
                msk * torch.sum(w * F.relu(self.neg_param + dist_neg_l), dim=-1),
                dim=-1,
            ) / (torch.sum(msk, dim=-1) + torch.finfo().eps)
        elif self.criterion == "cross_entropy":
            loss_neg = torch.sum(
                msk
                * torch.sum(
                    w * F.softplus(dist_neg_l / self.neg_param),
                    dim=-1,
                ),
                dim=-1,
            ) / (torch.sum(msk, dim=-1) + torch.finfo().eps)

        if do_log:
            min_dist_neg, argmin_dist_neg = torch.min(dist_neg_l, dim=-1)
            if known_c:
                err_neg = utils.masked_max(F.relu(min_dist_neg), msk, dim=-1)
                margin_neg = utils.masked_min(F.relu(-min_dist_neg), msk, dim=-1)
            else:
                obj_msk = argmin_dist_neg == dist_neg_l.shape[-1] - 1
                err_neg = utils.masked_max(F.relu(min_dist_neg), msk * ~obj_msk, dim=-1)
                err_obj = utils.masked_max(F.relu(min_dist_neg), msk * obj_msk, dim=-1)
                margin_neg = utils.masked_min(
                    F.relu(-min_dist_neg), msk * ~obj_msk, dim=-1
                )
                margin_obj = utils.masked_min(
                    F.relu(-min_dist_neg), msk * obj_msk, dim=-1
                )
                self.model().log_dict(
                    {
                        "obj/err": torch.mean(err_obj),
                        "obj/margin": torch.mean(margin_obj),
                    },
                )

            self.model().log_dict(
                {
                    "neg/loss": torch.mean(loss_neg),
                    "neg/err": torch.mean(err_neg),
                    "neg/margin": torch.mean(margin_neg),
                },
            )

        a_lk = torch.cat([a_l, F.normalize(a_k, dim=-1)], dim=-2)
        loss_var = torch.mean(torch.mean(a_lk, dim=-2) ** 2, dim=-1)

        if do_log:
            self.model().log_dict(
                {
                    "var/loss": torch.mean(loss_var),
                },
            )

        return self.balancer(
            {
                "pos": torch.mean(loss_pos),
                "neg": torch.mean(loss_neg),
                "var": torch.mean(loss_var),
            },
        )


class StaticBalancer(nn.Module):
    def __init__(
        self,
        weights: dict,
    ):
        super().__init__()
        self.weights = weights

    def forward(self, loss_dict):
        ret = 0
        for k, v in self.weights.items():
            ret += v * loss_dict[k]
        return ret


class CoVBalancer(nn.Module):
    def __init__(
        self,
        num_losses: int,
        decay: Optional[float] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.decay = decay
        self.eps = eps

        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("ema_dr", torch.zeros(()))
        self.register_buffer("ema_loss", torch.zeros(num_losses))
        self.register_buffer("ema_loss_ratio", torch.zeros(num_losses))
        self.register_buffer("var_nr", torch.zeros(num_losses))

    @torch.no_grad()
    def update(self, loss):
        self.step += 1
        decay = 1 / self.step if self.decay is None else self.decay

        self.ema_dr = (1 - decay) * self.ema_dr + decay * 1
        self.ema_loss = (1 - decay) * self.ema_loss + decay * loss
        self.ema_loss /= self.ema_dr
        loss_ratio = (loss + self.eps) / (self.ema_loss + self.eps)
        ema_loss_ratio = (1 - decay) * self.ema_loss_ratio + decay * loss_ratio
        ema_loss_ratio /= self.ema_dr
        self.var_nr = self.var_nr + (loss_ratio - self.ema_loss_ratio) * (
            loss_ratio - ema_loss_ratio
        )
        self.ema_loss_ratio = ema_loss_ratio

    @torch.no_grad()
    def compute(self):
        std_loss_ratio = torch.sqrt(self.var_nr / self.step + self.eps)
        cov_loss_ratio = std_loss_ratio / self.ema_loss_ratio
        self.w = cov_loss_ratio / torch.sum(cov_loss_ratio)

    def forward(self, loss_dict):
        loss = torch.stack(list(loss_dict.values()))
        self.update(loss)
        self.compute()
        self.model().log_dict(
            {("cov/" + k): self.w[i] for i, k in enumerate(loss_dict)}
        )
        return torch.sum(self.w * loss)
