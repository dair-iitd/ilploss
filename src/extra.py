from . import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List, Optional, Tuple, Union


class PartialAssignCEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.hparams = {}

    def forward(self, x):
        g = x != 0
        return (
            -F.one_hot((x - 1) * g, num_classes=self.num_classes)
            .float()
            .view(x.shape[0], -1)
        )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        output_dim: int,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ):
        def make_fc(out=False):
            layers = [
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim if out else hidden_dim),
            ]
            if batch_norm:
                layers = [nn.BatchNorm1d(hidden_dim)] + layers
            return nn.Sequential(*layers)

        super().__init__()
        self.inp = nn.Linear(input_dim, hidden_dim)
        self.num_blocks = num_blocks
        self.fcs = nn.ModuleList([make_fc() for _ in range(2 * num_blocks)])
        self.out = make_fc(True)

        self.hparams = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "output_dim": output_dim,
            "batch_norm": batch_norm,
            "dropout": dropout,
        }

    def forward(self, x):
        x = self.inp(x)
        for i in range(self.num_blocks):
            skip = x
            x = self.fcs[2 * i](x)
            x = self.fcs[2 * i + 1](x)
            x += skip
        x = self.out(x)
        return x


class MLPJointEncoder(MLP):
    def __init__(
        self,
        num_vars: int,
        num_constrs: int,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ):

        super().__init__(
            input_dim,
            hidden_dim,
            num_blocks,
            num_vars + num_constrs * (num_vars + 1),
            batch_norm,
            dropout,
        )
        self.num_vars = num_vars
        self.num_constrs = num_constrs

        self.hparams = {
            "num_vars": num_vars,
            "num_constrs": num_constrs,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "batch_norm": batch_norm,
            "dropout": dropout,
        }

    def forward(self, x):
        a, b, c = torch.split(
            super().forward(x),
            [self.num_constrs * self.num_vars, self.num_constrs, self.num_vars],
            dim=-1,
        )
        a = a.view(*a.shape[:-1], self.num_constrs, self.num_vars)
        breakpoint()
        return a, b, c


class CSPUniqueCEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.hparams = {}

    def forward(self, x):
        return 2 * x.float() - 1


class PartialAssignABEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.hparams = {}

    # FIXME: review code, or better, rewrite
    def forward(self, x):
        device = x.device
        x = x.view(x.shape[0], -1, self.num_classes)
        x = utils.zero_one_hot_decode(x)
        ret = torch.zeros(x.shape[0], 2, self.num_classes**3 + 1)
        g = x != 0
        c = torch.sum(g, dim=-1)
        x = F.one_hot((x - 1) * g, num_classes=self.num_classes).view(x.shape[0], -1)
        g = torch.repeat_interleave(g, self.num_classes, dim=-1)
        ret[:, 0, :-1] = -x * g
        ret[:, 0, -1] = c
        ret[:, 1, :-1] = (1 - x) * g
        return ret[:, :, :-1].to(device), ret[:, :, -1].to(device)


class ConcatABEncoder(nn.Module):
    def __init__(
        self,
        ab_encoders: List[nn.Module],
    ):
        super().__init__()
        self.ab_encoders = nn.ModuleList(ab_encoders)

        self.hparams = {}

    def forward(self, x):
        a, b = zip(*[enc(x) for enc in self.ab_encoders])
        return torch.cat(a, dim=-2), torch.cat(b, dim=-1)


class OneHotDecoder(nn.Module):
    def __init__(
        self,
        loss: Union[nn.Module, dict],
        num_classes: int,
    ):
        super().__init__()
        if isinstance(loss, dict):
            self.loss = instantiate_class(init=loss)
        else:
            self.loss = loss
        self.num_classes = num_classes

        self.hparams = {}
        if isinstance(loss, dict):
            self.hparams["loss"] = loss["init_args"]
        elif hasattr(loss, "hparams"):
            self.hparams["loss"] = loss.hparams

    # FIXME: review code, or better, rewrite
    def forward(self, zhat, y=None):
        self.zhat = zhat.view(-1, self.num_classes)
        pred = torch.argmax(self.zhat, dim=-1) + 1
        pred = pred.view(-1, (self.num_classes) ** 2)
        if y is None:
            return pred
        self.z = F.one_hot(y - 1, num_classes=self.num_classes)
        self.z = self.z.view(-1, self.num_classes)
        loss = self.loss(self.zhat, self.z)
        return pred, loss


class Identity(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.hparams = {}

    def forward(self, x):
        return x


class IntToOneHotDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.hparams = {}

    def forward(self, x):
        return F.one_hot(x - 1, num_classes=self.num_classes).view(x.shape[0], -1)


class RectifierModel(pl.LightningModule):
    def __init__(
        self,
        c_encoder: nn.Module,
        test_solver: nn.Module,
        num_vars: int,
        num_units: int,
    ):
        super().__init__()
        torch.backends.cudnn.benchmark = True

        self.c_encoder = c_encoder
        self.test_solver = test_solver
        self.num_vars = num_vars
        self.num_units = num_units
        self.linear = nn.Linear(num_vars, num_units)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.val_acc_bit = tm.Accuracy(
            subset_accuracy=False,
            compute_on_step=False,
        )
        self.val_acc_all = tm.Accuracy(
            subset_accuracy=True,
            compute_on_step=False,
        )
        self.val_acc_num = tm.Accuracy(
            subset_accuracy=True,
            compute_on_step=False,
        )

    def on_fit_start(self):
        dm = self.trainer.datamodule
        n, m = dm.box_shape
        self.num_classes = n * m

    def training_step(self, batch, batch_idx):
        i, x, y = batch

        c = self.c_encoder(x)

        yy = y[:, None, :].float()
        e = torch.eye(yy.shape[-1], device=yy.device)
        ybar = torch.cat([yy + e, yy - e], dim=-2)
        cc = c[:, None, :]
        ymsk = torch.all((ybar >= 0) & (ybar <= 1), dim=-1)
        ymsk *= torch.sum(cc * ybar, dim=-1) <= torch.sum(cc * yy, dim=-1)

        yhat_p = 1 - torch.sum(F.relu(self.linear(yy)), dim=-1)
        yhat_n = 1 - torch.sum(F.relu(self.linear(ybar)), dim=-1)
        loss_p = torch.mean(self.criterion(yhat_p, torch.ones_like(yhat_p)))
        loss_n = torch.mean(self.criterion(yhat_n, torch.zeros_like(yhat_n)) * ymsk)
        loss = loss_p + loss_n

        self.log_dict(
            {
                "train/loss/pos": loss_p,
                "train/loss/neg": loss_n,
                "train/loss": loss,
            }
        )

        return loss

    def validation_step(self, batch, batch_idx):
        i, x, y = batch

        c = self.c_encoder(x)

        device = x.device
        l = torch.arange(2**self.num_units, device=device)
        r = torch.arange(self.num_units, device=device)
        enum = (l[:, None] & (2**r)) > 0
        a = torch.sum(self.linear.weight[None, :, :] * enum[:, :, None], dim=-2)
        b = 1 - torch.sum(self.linear.bias[None, :] * enum[:, :], dim=-1)

        yhat = self.test_solver(
            a.expand(x.shape[-1], -1, -1),
            b.expand(x.shape[-1], -1),
            c,
            None,
        )

        yhat = yhat.long()
        self.val_acc_bit(yhat, y)
        self.val_acc_all(yhat, y)

        yhat = yhat.view(-1, self.num_classes)
        y = y.view(-1, self.num_classes)
        self.val_acc_num(yhat, y)

        self.log_dict(
            {
                "val/acc/bit": self.val_acc_bit,
                "val/acc/all": self.val_acc_all,
                "val/acc/num": self.val_acc_num,
            }
        )


class ProjILPLossTerms(nn.Module):
    def __init__(
        self,
        num_constrs: int,
    ):
        super().__init__()

        self.gain = nn.Parameter(torch.ones(num_constrs), requires_grad=False)

        self.hparams = {}

    def forward(self, inp, y):
        a, b, c, h = inp
        a = a.transpose(-1, -2)
        b = b[:, None, :]
        norm = torch.linalg.vector_norm(a, dim=-2, keepdim=True)
        a = a / norm
        b = b / norm
        y = y.float()

        pos = y[:, None, :]
        dist_p = pos @ a + b
        with torch.no_grad():
            proj = (y[:, :, None] - dist_p * a).transpose(-1, -2)
            floor = torch.floor(proj)
            frac = proj - floor
            rand = torch.rand_like(frac) <= frac
            neg = floor + rand
            diff = torch.abs(proj - neg)
            mx = torch.max(diff, dim=-1)
            neg[diff == mx] += torch.sign(proj - neg)
        dist_n = neg @ a + b

        cls_loss_p = F.relu(0.1 - self.gain * dist_p)
        cls_loss_n = F.relu(0.1 + self.gain * dist_n)
        w = F.softmax(-cls_loss_n, dim=-1)
        mw = torch.mean(w, dim=-2)[:, None, :]
        self.loss_p = torch.mean(torch.sum(mw * cls_loss_p, dim=-1))
        self.loss_n = torch.mean(torch.sum(w * cls_loss_n, dim=-1))
        self.loss_v = torch.mean(torch.sum(torch.sum(mw * a, dim=-1) ** 2, dim=-1))
        self.loss_g = torch.mean(torch.sum(mw * self.gain**2, dim=-1))

        return {
            "pos": self.loss_p,
            "neg": self.loss_n,
            "var": self.loss_v,
            "gain": self.loss_g,
        }


class PlaceboSolver(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.hparams = {}

    def forward(self, *args):
        return args


class NearestNbrNegSampler(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.hparams = {}

    def forward(self, inp, y):
        a, b, c, l, u, h = inp
        y = y[:, None, :]
        e = torch.eye(y.shape[-1], device=y.device)
        n = torch.cat([y + e, y - e], dim=-2)
        nmsk = torch.all((l[:, None, :] <= n) & (n <= u[:, None, :]), dim=-1)
        return n, nmsk


class ProjNegSampler(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.hparams = {}

    def forward(self, a, b, c, y):
        y = y[:, None, :]
        neg = torch.round(y + (b - utils.dot(a, y, dim=-1))[:, :, None] * a)
        # e = torch.eye(y.shape[-1], device=y.device)
        # return torch.cat([neg, y+e, y-e], dim=-2)
        return neg


class ILPLoss(nn.Module):
    def __init__(
        self,
        neg_sampler: nn.Module,
        cls_loss: nn.Module,
        pos_label: float = 1.0,
        neg_label: float = 0.0,
        min_: Callable = utils.ws_softmin,
        tau: float = 1.0,
        neg_pos_ratio: float = 1.0,
    ):
        super().__init__()
        self.neg_sampler = neg_sampler
        self.cls_loss = cls_loss
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.min_ = min_
        self.tau = tau
        self.neg_pos_ratio = neg_pos_ratio

        self.hparams = {
            "neg_sampler": utils.hparams(neg_sampler),
            "cls_loss": utils.hparams(cls_loss),
            "neg_pos_ratio": utils.hparams(neg_pos_ratio),
            "tau": utils.hparams(tau),
        }

    def forward(self, inp, y):
        a, b, c, l, u, h = inp
        # l = l.float() - 0.7
        # u = u.float() + 0.7
        a, b = utils.add_lu_constrs(a, b, l, u)
        y = y.float()

        pos = y[:, None, :]
        # with torch.no_grad():
        #     if isinstance(self.neg_sampler, SolveNegSampler):
        #         neg = self.neg_sampler(inp, y)
        #     else:
        #         neg = self.neg_sampler(a, b, c, y)
        neg = self.neg_sampler(a, b, c, y).detach()

        a_t = a.transpose(-1, -2)
        b = b[:, None, :]
        dist_p = b - pos @ a_t
        dist_n = b - neg @ a_t

        label_p = torch.tensor(self.pos_label).expand(*dist_p.shape)
        label_n = torch.tensor(self.neg_label).expand(*dist_n.shape)
        cls_loss_p = self.cls_loss(dist_p, label_p)
        cls_loss_n = self.cls_loss(dist_n, label_n)
        self.loss_p = torch.mean(torch.sum(cls_loss_p, dim=-1))
        msk = torch.any(neg != pos, dim=-1)
        msk = msk * (dist_p.detach()[:, 0, :] >= 0)
        # msk = msk * torch.cat([dist_p.detach()[:,0,:] >= 0,
        #     torch.ones(a.shape[0],2*a.shape[-1])], dim=-1)
        self.loss_n = torch.mean(self.min_(cls_loss_n, self.tau, dim=-1) * msk)

        e = torch.eye(a.shape[-2], device=a.device)
        loss_aux = torch.mean(torch.sum((a @ a_t - e) ** 2, dim=-1))

        return (self.loss_p + self.neg_pos_ratio * self.loss_n) / (
            1 + self.neg_pos_ratio
        ) + loss_aux


class OldDisjointEncoder(nn.Module):
    def __init__(
        self,
        ab_encoder: nn.Module,
        c_encoder: nn.Module,
        lu_encoder: nn.Module,
    ):
        super().__init__()
        self.ab_encoder = ab_encoder
        self.c_encoder = c_encoder
        self.lu_encoder = lu_encoder

        self.hparams = {
            "ab_encoder": utils.hparams(ab_encoder),
            "c_encoder": utils.hparams(c_encoder),
            "lu_encoder": utils.hparams(lu_encoder),
        }

    def forward(self, x):
        a, b = self.ab_encoder(x)
        c = self.c_encoder(x)
        l, u = self.lu_encoder(x)
        return a, b, c, l, u


class StaticWeightedLoss(nn.Module):
    def __init__(
        self,
        criterion: nn.Module,
        weights: dict,
    ):
        super().__init__()
        self.criterion = criterion
        self.weights = weights

    def forward(self, yhat, y):
        loss_terms = self.criterion(yhat, y)
        ret = 0
        for k, v in self.weights.items():
            ret += v * loss_terms[k]
        return ret


class CoVWeightedLoss(nn.Module):
    def __init__(
        self,
        criterion: nn.Module,
        num_losses: int,
        decay: Optional[float] = None,
        epsilon: float = 1e-9,
    ):
        super().__init__()
        self.criterion = criterion
        self.decay = decay
        self.epsilon = epsilon

        self.step = 0
        self.ema_dr = 0
        self.ema_loss = 0
        self.ema_loss_ratio = 0
        self.var_nr = 0

    @torch.no_grad()
    def update(self, loss):
        self.step += 1
        decay = 1 / self.step if self.decay is None else self.decay

        self.ema_dr = (1 - decay) * self.ema_dr + decay * 1
        self.ema_loss = (1 - decay) * self.ema_loss + decay * loss
        self.ema_loss /= self.ema_dr
        loss_ratio = loss / self.ema_loss
        ema_loss_ratio = (1 - decay) * self.ema_loss_ratio + decay * loss_ratio
        ema_loss_ratio /= self.ema_dr
        self.var_nr = self.var_nr + (loss_ratio - self.ema_loss_ratio) * (
            loss_ratio - ema_loss_ratio
        )
        self.ema_loss_ratio = ema_loss_ratio

    @torch.no_grad()
    def compute(self):
        std_loss_ratio = torch.sqrt(self.var_nr / self.step + self.epsilon)
        cov_loss_ratio = std_loss_ratio / self.ema_loss_ratio
        self.w = cov_loss_ratio / torch.sum(cov_loss_ratio)

    def forward(self, yhat, y):
        loss_terms = self.criterion(yhat, y)
        loss = torch.stack(list(loss_terms.values()))
        self.update(loss)
        self.compute()
        with logging_redirect_tqdm():
            logger.debug(f"{self.w}")
        return torch.sum(self.w * loss)


class CONLoss(nn.Module):
    def __init__(
        self,
        tau: float = 1.0,
    ):
        super().__init__()

        self.tau = tau

        self.hparams = {
            "tau": utils.hparams(tau),
        }

    def min(self, x, dim):
        return -self.tau * torch.logsumexp(-x / self.tau, dim=dim)

    def forward(self, a, b, c, pos, neg):
        norm_a = torch.linalg.vector_norm(a, dim=-1)
        a = a / norm_a[:, :, None]
        b = b / norm_a
        a_t = a.transpose(-1, -2)
        b_t = b[:, None, :]

        norm_c = torch.linalg.vector_norm(c, dim=-1)
        c = c / norm_c[:, None]
        c_t = c[:, :, None]

        dist_pos = pos @ a_t + b_t
        dist_neg = neg @ a_t + b_t
        dist_obj = (pos - neg) @ c_t

        loss_pos = torch.sum(F.relu(-dist_pos), dim=-1)
        loss_neg = self.min(F.relu(dist_neg), dim=-1)
        loss_obj = F.relu(dist_obj)[:, :, 0]

        # msk = torch.all(dist_pos >= 0, dim=-1)
        # loss = loss_pos * ~msk + loss_neg * msk + loss_obj * msk
        loss = loss_pos + (loss_neg + loss_obj) * (loss_pos == 0)

        models.model.log_dict(
            {
                "train/loss/pos": torch.mean(loss_pos),
                "train/loss/neg": torch.mean(loss_neg),
                "train/loss/obj": torch.mean(loss_obj),
                "train/loss/net": torch.mean(loss),
                "train/feas": torch.mean(torch.sum(loss_pos == 0, dim=-1).float()),
            }
        )


class DisjointEncoder(nn.Module):
    def __init__(
        self,
        ab_encoders: List[nn.Module],
        c_encoder: nn.Module,
    ):
        super().__init__()
        self.ab_encoders = nn.ModuleList(ab_encoders)
        self.c_encoder = c_encoder

    def forward(self, x):
        a_list, b_list = zip(*[enc(x) for enc in self.ab_encoders])
        a = torch.cat(a_list, dim=-2)
        b = torch.cat(b_list, dim=-1)
        c = self.c_encoder(x)
        return a, b, c


class VisualSudokuDataset(Dataset):
    def __init__(
        self,
        x_img: torch.Tensor,
        y: torch.Tensor,
        x_sym: torch.Tensor,
    ):
        super().__init__()
        self.x_img = x_img
        self.y = y
        self.x_sym = x_sym

        self.transform = torchvision.transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (
            idx,
            torch.stack(
                [
                    self.transform(Image.fromarray(img.numpy(), mode="L"))
                    for img in self.x_img[idx]
                ]
            ),
            self.y[idx],
            self.x_sym[idx],
        )


def sudoku_to_html(
    n,
    m,
    yhat,
    x,
    y,
    hc="-",
    vc="|",
    cc="|",
    sc=" ",
    nc="\n",
    eq_x_dec=["", ""],
    eq_y_dec=['<font color="green">', "</font>"],
    neq_y_dec=['<font color="red">', "</font>"],
):
    x = x.view(m, n, n, m)
    yhat = yhat.view(m, n, n, m)
    y = y.view(m, n, n, m)
    mx = int(math.log10(n * m)) + 1
    bl = []
    for b in range(m):
        rl = []
        for r in range(n):
            sl = []
            for s in range(n):
                cl = [""]
                for c in range(m):
                    if yhat[b, r, s, c] == x[b, r, s, c]:
                        dec = eq_x_dec
                    elif yhat[b, r, s, c] == y[b, r, s, c]:
                        dec = eq_y_dec
                    else:
                        dec = neq_y_dec
                    cl.append(f"{dec[0]}{yhat[b,r,s,c].item(): <{mx}}{dec[1]}")
                cl.append("")
                sl.append(sc.join(cl))
            rl.append(vc.join(sl))
        bl.append(nc.join(rl))
    bc = nc + cc.join([hc.join([""] + [hc * mx] * m + [""])] * n) + nc
    return f"<pre>{bc.join(bl)}</pre>"


class BCECriterion(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.hparams = {}

    def forward(self, dist, pos):
        if pos:
            target = torch.ones_like(dist)
        else:
            target = torch.zeros_like(dist)
        return F.binary_cross_entropy_with_logits(dist, target, reduction="none")


class HingeCriterion(nn.Module):
    def __init__(
        self,
        margin: float = 0.0,
    ):
        super().__init__()
        self.margin = margin

        self.hparams = {
            "margin": utils.hparams(margin),
        }

    def forward(self, dist, pos):
        if pos:
            return F.relu(self.margin - dist)
        else:
            return F.relu(self.margin + dist)


class BitPartialAssignCEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.hparams = {}

    def forward(self, x):
        c = 1 - 2 * x
        xx = x.view(x.shape[0], -1, self.num_classes)
        g = utils.zero_one_hot_decode(xx) != 0
        g = g[:, :, None].expand(-1, -1, self.num_classes).reshape(g.shape[0], -1)
        return c * g


class LPLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, a, b, c, y):
        """
        a: batch x constrs x vars
        b: batch x constrs
        c: batch x vars
        y: batch x vars
        """
        a = torch.cat([a, -c[:, None, :]], dim=-2)
        b = torch.cat([b, torch.sum(c * y, dim=-1)[:, None]], dim=-1)

        norm = torch.linalg.norm(a, dim=-1)
        a = a / norm[:, :, None]
        b = b / norm
        """
        norm: batch x constrs
        """

        dist = torch.sum(a * y[:, None, :], dim=-1) + b
        """
        dist: batch x constrs
        """

        e = torch.eye(a.shape[-1], device=a.device)
        ray = torch.cat([a, e[None, :, :].expand(a.shape[0], -1, -1)], dim=-2)
        ray = torch.cat([ray, -ray], dim=-2)

        r = -dist[:, None, :] / (ray @ a.transpose(-1, -2))
        r = r.clone()
        r[r < 0] = 1e12
        w = F.one_hot(torch.argmin(r, dim=-1), num_classes=r.shape[-1])
        # w = F.softmax(-r, dim=-1)
        """
        dir_: batch x dirs x vars
        r: batch x dirs x constrs
        w: batch x dirs x constrs
        """

        # loss_sat = torch.sum(F.relu(-dist), dim=-1)
        loss_sat = torch.max(F.relu(-dist), dim=-1)[0]
        # loss_opt = dist.shape[-1] * torch.mean(torch.sum(w * r, dim=-1), dim=-1)
        # loss_opt = dist.shape[-1] * torch.max(torch.sum(w * r, dim=-1), dim=-1)[0]
        loss_opt = torch.max(torch.sum(w * r, dim=-1), dim=-1)[0]
        loss_var = torch.sum(torch.sum(a, dim=-2) ** 2, dim=-1)
        """
        loss_sat: batch
        loss_opt: batch
        loss_var: batch
        """

        with torch.no_grad():
            err_sat = torch.max(F.relu(-dist), dim=-1)[0]
            w = f.one_hot(torch.argmin(r, dim=-1), num_classes=r.shape[-1])
            err_opt = torch.max(torch.sum(w * r, dim=-1), dim=-1)[0]

        models.model.log_dict(
            {
                "train/loss/sat": torch.mean(loss_sat, dim=-1),
                "train/loss/opt": torch.mean(loss_opt, dim=-1),
                "train/loss/var": torch.mean(loss_var, dim=-1),
                "train/err/sat": torch.mean(err_sat, dim=-1),
                "train/err/opt": torch.mean(err_opt, dim=-1),
            },
        )

        return torch.mean(1000 * loss_sat + loss_opt + loss_var, dim=-1)


class HingeLoss(nn.Module):
    def __init__(
        self,
        pos_margin: float,
        neg_margin: float,
        reduction: str = "mean",
    ):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.reduction = reduction

        self.hparams = {
            "pos_margin": utils.hparams(pos_margin),
            "neg_margin": utils.hparams(neg_margin),
        }

    def forward(self, yhat, y):
        pos_pre_loss = self.pos_margin - yhat
        neg_pre_loss = self.neg_margin + yhat
        loss = F.relu(y * pos_pre_loss + (1 - y) * neg_pre_loss)
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "mean":
            return torch.mean(loss)
        else:
            raise ValueError(f"Reduction must be one of none | sum | mean.")

        return torch.mean(loss)


class BitLUEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        margin: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin

        self.hparams = {
            "margin": utils.hparams(margin),
        }

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.num_classes)
        l = x.clone()
        u = x.clone()
        msk = (utils.zero_one_hot_decode(x) == 0)[:, :, None]
        msk = msk.expand(-1, -1, self.num_classes)
        l[msk] = 0
        u[msk] = 1
        l, u = l.view(l.shape[0], -1), u.view(u.shape[0], -1)
        return l - self.margin, u + self.margin


class StaticBitLUEncoder(nn.Module):
    def __init__(
        self,
        num_vars: int,
        margin: float = 0.5,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.margin = margin

        self.hparams = {
            "margin": utils.hparams(margin),
        }

    def forward(self, x):
        return (
            torch.zeros(x.shape[0], self.num_vars, device=x.device) - self.margin,
            torch.ones(x.shape[0], self.num_vars, device=x.device) + self.margin,
        )


class NumLUEncoder(nn.Module):
    def __init__(
        self,
        lb: int,
        ub: int,
        margin: float = 0.5,
    ):
        super().__init__()
        self.lb = lb
        self.ub = ub
        self.margin = margin

        self.hparams = {
            "lb": lb,
            "ub": ub,
            "margin": utils.hparams(margin),
        }

    def forward(self, x):
        l = x.clone()
        u = x.clone()
        msk = x == 0
        l[msk] = self.lb
        u[msk] = self.ub
        return l - self.margin, u + self.margin
