from . import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from IPython.core.debugger import Pdb
from typing import Optional

logger = logging.getLogger(__name__)


class RectifierCore(nn.Module):
    def __init__(
        self,
        rect_ab_encoder: nn.Module,
        c_encoder: nn.Module,
        sampler: nn.Module,
        balancer: nn.Module,
        known_ab_encoder: nn.Module,
        label_pos: float = 1.0,
        label_neg: float = 0.0,
    ):
        super().__init__()

        self.rect_ab_encoder = rect_ab_encoder
        self.c_encoder = c_encoder
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.balancer = balancer
        self.label_pos = label_pos
        self.label_neg = label_neg
        self.known_ab_encoder = known_ab_encoder
        self.sampler = sampler

        self.hparams = {
            "rect_ab_encoder": utils.hparams(rect_ab_encoder),
            "known_ab_encoder": utils.hparams(known_ab_encoder),
            "c_encoder": utils.hparams(c_encoder),
            "balancer": utils.hparams(balancer),
            "sampler": utils.hparams(sampler),
        }

    def forward(self, i, x, y=None):
        c = self.c_encoder(x)
        a_k, b_k = self.known_ab_encoder(x)
        a_l, b_l = self.rect_ab_encoder(x)
        if not self.training:
            return torch.cat([a_l, a_k], dim=-2), torch.cat([b_l, b_k], dim=-1), c
        pos = y
        with torch.no_grad():
            neg = self.sampler([a_l, a_k], [b_l, b_k], c, y)

        yhat_pos = self.rect_ab_encoder.predict(pos)
        yhat_neg = self.rect_ab_encoder.predict(neg)

        target_pos = torch.empty_like(yhat_pos).fill_(self.label_pos)
        target_neg = torch.empty_like(yhat_neg).fill_(self.label_neg)

        fdist_neg_k = neg @ a_k.transpose(-1, -2) + b_k[..., None, :]
        msk = torch.any(neg != pos[..., None, :], dim=-1)
        msk = msk * torch.all(fdist_neg_k >= 0, dim=-1)

        loss_neg = (
            (msk * self.criterion(yhat_neg, target_neg)).sum(dim=-1) / msk.sum(dim=-1)
        ).mean()
        loss_pos = torch.mean(self.criterion(yhat_pos, target_pos), dim=-1)

        # print("######################################")
        # incorrect = yhat_neg.argmax(dim=-1)
        # print(incorrect)
        # for i in range(incorrect.size(0)):
        #    print('--------------------------------------------------------')
        #    which_cell = (neg[i,incorrect[i]].view(-1,4) != pos[i].view(-1,4)).any(dim=-1)
        #    print(neg[i,incorrect[i]].view(-1,4)[which_cell])
        #    print(pos[i].view(-1,4)[which_cell])
        #    print(neg[i, incorrect[i]].view(4,4,4).argmax(-1)+1)
        #    print(pos[i].view(4,4,4).argmax(-1)+1)
        #    print('--------------------------------------------------------')

        # print("######################################")

        acc_pos = yhat_pos > 0
        is_correctly_classified = ((yhat_neg <= 0).long() + 1 - msk.long()) > 0
        acc_neg = torch.all(is_correctly_classified, dim=-1)
        acc_neg_frac = is_correctly_classified.float().mean(dim=-1)
        acc = acc_pos & acc_neg

        ######cross check using a_l and b_l
        # dist_pos_l = torch.sum(a_l * pos[..., None, :], dim=-1) + b_l
        # dist_neg_l = neg @ a_l.transpose(-1, -2) + b_l[..., None, :]
        # ab_acc_pos = torch.mean((dist_pos_l > 0).float())
        # ab_acc_neg = torch.all(
        #    torch.any(dist_neg_l < 0, dim=-1),
        #    dim=-1
        # ).float().mean()
        # ab_acc_neg_frac = torch.mean(
        #    (torch.any(dist_neg_l < 0, dim=-1) ).float(),
        #    dim=-1,
        # ).mean()
        # print("AB accuracies:", ab_acc_pos, ab_acc_neg, ab_acc_neg_frac)
        # print("Rectifier MLP  accuracies:", acc_pos.float().mean(), acc_neg.float().mean(), acc_neg_frac.mean())
        # Pdb().set_trace()
        #######

        self.model().log_dict(
            {
                "acc/pos": torch.mean(acc_pos.float()),
                "acc/neg": torch.mean(acc_neg.float()),
                "acc/neg_frac": torch.mean(acc_neg_frac.float()),
                "acc": torch.mean(acc.float()),
            },
        )

        self.model().log_dict(
            {
                "train/loss/pos": loss_pos,
                "train/loss/neg": loss_neg,
            }
        )

        return self.balancer(
            {
                "pos": loss_pos,
                "neg": loss_neg,
            }
        )


class SolverFreeCore(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        known_ab_encoder: nn.Module,
        sampler: nn.Module,
        criterion: nn.Module,
        cost_train_criterion: Optional[nn.Module] = None,
        cost_loss_wt: Optional[float] = 0.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.known_ab_encoder = known_ab_encoder
        self.sampler = sampler
        self.criterion = criterion
        self.cost_train_criterion = cost_train_criterion
        self.cost_loss_wt = cost_loss_wt

        self.hparams = {
            "encoder": utils.hparams(encoder),
            "known_ab_encoder": utils.hparams(known_ab_encoder),
            "sampler": utils.hparams(sampler),
            "criterion": utils.hparams(criterion),
            "cost_loss_wt": cost_loss_wt,
        }

    def forward(self, i, x, y=None):
        a_l, b_l, c = self.encoder(x)
        a_k, b_k = self.known_ab_encoder(x)
        if not self.training:
            return torch.cat([a_l, a_k], dim=-2), torch.cat([b_l, b_k], dim=-1), c

        pos = y
        with torch.no_grad():
            neg = self.sampler([a_l, a_k], [b_l, b_k], c, y)

        solverfree_loss = self.criterion([a_l, a_k], [b_l, b_k], c, pos, neg)
        # return loss
        cost_loss = 0.0
        if self.cost_loss_wt > 0:
            cost_loss = self.cost_train_criterion(-c, y)
        #
        return (
            1.0 - self.cost_loss_wt
        ) * solverfree_loss + self.cost_loss_wt * cost_loss
