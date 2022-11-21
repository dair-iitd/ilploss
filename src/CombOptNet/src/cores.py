from . import models, utils

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MySolverCore(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        known_ab_encoder: nn.Module,
        solver: nn.Module,
        criterion: nn.Module,
        cost_train_criterion: Optional[nn.Module] = None,
        cost_loss_wt: Optional[float] = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.known_ab_encoder = known_ab_encoder
        self.solver = solver
        self.criterion = criterion
        self.cost_train_criterion = cost_train_criterion
        self.cost_loss_wt = cost_loss_wt

        self.hparams = {
            "encoder": utils.hparams(encoder),
            "known_ab_encoder": utils.hparams(known_ab_encoder),
            "solver": utils.hparams(solver),
            "criterion": utils.hparams(criterion),
            "cost_loss_wt": cost_loss_wt,
        }

    def forward(self, i, x, y=None):
        a_l, b_l, c = self.encoder(x)
        a_k, b_k = self.known_ab_encoder(x)
        if not self.training:
            return torch.cat([a_l, a_k], dim=-2), torch.cat([b_l, b_k], dim=-1), c

        solver_loss = 0.0
        if 1.0 - self.cost_loss_wt > 0:
            yhat, status = self.solver(a_l, a_k, b_l, b_k, c, None)
            solver_loss = self.criterion(yhat, y)

        cost_loss = 0.0
        if self.cost_loss_wt > 0:
            cost_loss = self.cost_train_criterion(-c, y)

        return (1.0 - self.cost_loss_wt) * solver_loss + self.cost_loss_wt * cost_loss


class SolverCore(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        solver: nn.Module,
        criterion: nn.Module,
        known_ab_encoder: Optional[nn.Module] = None,
        cost_train_criterion: Optional[nn.Module] = None,
        cost_loss_wt: Optional[float] = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.solver = solver
        self.criterion = criterion
        self.known_ab_encoder = known_ab_encoder
        self.cost_train_criterion = cost_train_criterion
        self.cost_loss_wt = cost_loss_wt

        self.hparams = {
            "encoder": utils.hparams(encoder),
            "solver": utils.hparams(solver),
            "criterion": utils.hparams(criterion),
            "known_ab_encoder": utils.hparams(known_ab_encoder),
            "cost_loss_wt": cost_loss_wt,
        }

    def forward(self, i, x, y=None):
        a, b, c = self.encoder(x)
        if self.known_ab_encoder is not None:
            a_k, b_k = self.known_ab_encoder(x)
            a = torch.cat([a, a_k], dim=-2)
            b = torch.cat([b, b_k], dim=-1)

        if y is None:
            return a, b, c

        solver_loss = 0.0
        if 1.0 - self.cost_loss_wt > 0:
            yhat, status = self.solver(a, b, c, None)
            solver_loss = self.criterion(yhat, y)

        cost_loss = 0.0
        if self.cost_loss_wt > 0:
            cost_loss = self.cost_train_criterion(-c, y)
        #
        return (1.0 - self.cost_loss_wt) * solver_loss + self.cost_loss_wt * cost_loss
