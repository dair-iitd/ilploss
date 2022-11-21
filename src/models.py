import os
from . import utils
from .CombOptNet.src import models, ilp
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import instantiate_class
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from torchvision.utils import make_grid

import collections
import logging
import math
from IPython.core.debugger import Pdb
logger = logging.getLogger(__name__)


class SudokuModel(models.DecoderFreeModel):
    def __init__(
        self,
        num_classes: int,
        core: nn.Module,
        solver: nn.Module,
        optimizer: dict,
        lr_scheduler: dict,
        schedulers: dict = {},
        x_key: str = "x",
        y_key: str = "y",
        hint: str = "none",
        cache_init: dict = {"class_path": "torch.nn.init.zeros_"},
    ):
        super().__init__(
            x_key=x_key,
            y_key=y_key,
            core=core,
            solver=solver,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            schedulers=schedulers,
            hint=hint,
            cache_init=cache_init,
        )

        self.num_classes = num_classes

        if x_key == "query/img":
            self.metrics.update(
                {
                    "train/acc/pre": tm.Accuracy(subset_accuracy=False),
                    "val/acc/pre": tm.Accuracy(subset_accuracy=False),
                    "test/acc/pre": tm.Accuracy(subset_accuracy=False),
                },
            )
        if y_key == "target/bit":
            self.metrics.update(
                {
                    "train/acc/num": tm.Accuracy(
                        subset_accuracy=True,
                        mdmc_average="samplewise",
                    ),
                    "val/acc/num": tm.Accuracy(
                        subset_accuracy=True,
                        mdmc_average="samplewise",
                    ),
                    "test/acc/num": tm.Accuracy(
                        subset_accuracy=True,
                        mdmc_average="samplewise",
                    ),
                },
            )
        self.my_hparams.update({})
        self.hparam_metrics += ["val/acc/num", "val/status/optimal"]

    def validation_step_log(self, batch, batch_idx, dataloader_idx, a, b, c, yhat):
        r = "train" if dataloader_idx == 0 else "val"

        if self.x_key == "query/img":
            self.metrics[f"{r}/acc/pre"](
                torch.argmin(c.contiguous().view(-1, self.num_classes), dim=-1)[
                    batch["query/num"].long().view(-1) != 0
                ],
                batch["query/num"].long().view(-1)[batch["query/num"].view(-1) != 0],
            )

        lb = torch.minimum(torch.min(yhat).long(), batch[self.y_key].long())
        self.metrics[f"{r}/acc/idv"](yhat.long() - lb, batch[self.y_key].long() - lb)
        self.metrics[f"{r}/acc/all"](yhat.long() - lb, batch[self.y_key].long() - lb)

        if self.y_key == "target/bit":
            self.metrics[f"{r}/acc/num"](
                yhat.long().view(-1, self.num_classes),
                batch["target/bit"].long().view(-1, self.num_classes),
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if not self.should_validate():
            return

        i, x, y = batch["idx"], batch[self.x_key], batch[self.y_key]
        a, b, c = self.core(i, x)

        # TODO: implement as a plugable module
        if self.hint == "none":
            h = None
        elif self.hint == "gold":
            h = y
        elif self.hint == "cache":
            h = self.cache[dataloader_idx][i]
        elif self.hint == "pre":
            h = torch.flatten(
                F.one_hot(
                    torch.argmin(c.contiguous().view(-1, self.num_classes), dim=-1),
                    num_classes=self.num_classes,
                ),
                -2,
                -1,
            ).float()

        yhat, status = self.solver(a, b, c, h)

        if self.hint == "cache":
            self.cache[dataloader_idx][i] = yhat.detach().cpu()

        if batch_idx == 0 and dataloader_idx == 1:
            self.val_save = batch, a, b, c, yhat

        self.validation_step_log(batch, batch_idx, dataloader_idx, a, b, c, yhat)

        return status

    def validation_epoch_end(self, outputs):
        val_metrics = super().validation_epoch_end(outputs)
        if not self.should_validate():
            logger.info(f"skip validation for {self.current_epoch}")
            return

        batch, a, b, c, yhat = self.val_save
        x, y = batch[self.x_key], batch[self.y_key]

        if self.x_key == "query/img":
            try:
                tb = self.logger.experiment
                tb.add_figure(
                    "val/query/img",
                    sns.heatmap(
                        make_grid(batch["query/img"][0], nrow=self.num_classes)[0]
                        .detach()
                        .cpu(),
                        cmap="gray",
                    ).get_figure(),
                    self.global_step,
                )
            except AttributeError:
                pass

        if self.y_key == "target/bit":
            logger.info("acc/num ")
            logger.info(
                " \t"
                + " | ".join(
                    [
                        f"train: {val_metrics['train/acc/num']:.4f}",
                        f"val: {val_metrics['val/acc/num']:.4f}",
                    ]
                )
            )

    def test_step_log(self, batch, batch_idx, dataloader_idx, a, b, c, yhat):
        if self.x_key == "query/img":
            self.metrics[f"test/acc/pre"](
                torch.argmin(c.contiguous().view(-1, self.num_classes), dim=-1)[
                    batch["query/num"].long().view(-1) != 0
                ],
                batch["query/num"].long().view(-1)[batch["query/num"].view(-1) != 0],
            )

        lb = torch.minimum(torch.min(yhat).long(), batch[self.y_key].long())
        self.metrics[f"test/acc/idv"](yhat.long() - lb, batch[self.y_key].long() - lb)
        cur_acc_all = self.metrics[f"test/acc/all"](
            yhat.long() - lb, batch[self.y_key].long() - lb
        ).item()
        net_acc_all = self.metrics[f"test/acc/all"].compute().item()
        logger.info(f"{batch_idx=} {cur_acc_all=} {net_acc_all=}")

        if self.y_key == "target/bit":
            self.metrics[f"test/acc/num"](
                yhat.long().view(-1, self.num_classes),
                batch["target/bit"].long().view(-1, self.num_classes),
            )

    def test_epoch_end(self, outputs):
        test_metrics = super().test_epoch_end(outputs)

        batch, a, b, c, yhat = self.test_save
        x, y = batch[self.x_key], batch[self.y_key]

        if self.x_key == "query/img":
            try:
                tb = self.logger.experiment
                tb.add_figure(
                    "test/query/img",
                    sns.heatmap(
                        make_grid(batch["query/img"][0], nrow=self.num_classes)[0]
                        .detach()
                        .cpu(),
                        cmap="gray",
                    ).get_figure(),
                    self.global_step,
                )
            except AttributeError:
                pass

        if self.y_key == "target/bit":
            logger.info("acc/num ")
            logger.info(
                " \t"
                + " | ".join(
                    [
                        f"test: {test_metrics['test/acc/num']:.4f}",
                    ]
                )
            )


class GMModel(models.DecoderFreeModel):
    def __init__(
        self,
        core: nn.Module,
        solver: nn.Module,
        optimizer: dict,
        lr_scheduler: dict,
        schedulers: dict = {},
        hint: str = "none",
        cache_init: dict = {"class_path": "torch.nn.init.zeros_"},
        reset_on_temp_change: bool = False,
    ):
        super().__init__(
            x_key="x",
            y_key="y",
            core=core,
            solver=solver,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            schedulers=schedulers,
            hint=hint,
            cache_init=cache_init,
            reset_on_temp_change = reset_on_temp_change,
        )

        self.metrics.update(
            {
                "train/acc/num": tm.Accuracy(
                    subset_accuracy=True,
                    mdmc_average="samplewise",
                ),
                "val/acc/num": tm.Accuracy(
                    subset_accuracy=True,
                    mdmc_average="samplewise",
                ),
                "test/acc/num": tm.Accuracy(
                    subset_accuracy=True,
                    mdmc_average="samplewise",
                ),
                "train/acc/pre": tm.Accuracy(
                    subset_accuracy=False,
                    #mdmc_average="samplewise",
                ),
                "val/acc/pre": tm.Accuracy(
                    subset_accuracy=False,
                    #mdmc_average="samplewise",
                ),
                "test/acc/pre": tm.Accuracy(
                    subset_accuracy=False,
                    #mdmc_average="samplewise",
                ),
                "train/constraint_satisfied": tm.Accuracy(),
                "val/constraint_satisfied": tm.Accuracy(),
                "test/constraint_satisfied": tm.Accuracy(),
            }
        )
        self.my_hparams.update({})
        self.hparam_metrics += ["val/acc/num", "val/status/optimal"]

        self.scratch = {}

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if optimizer_idx != 0:
            return None
        if self.config['log_idx']:
            with open(os.path.join(self.logger.log_dir,'train_permutations.csv'), 'a') as fh:
                to_log = torch.cat([batch['x']['perm1'],batch['x']['perm2'], batch['x']['kps']], dim=1)
                print('\n'.join([','.join(x) for x in to_log.cpu().numpy().astype(int).astype(str).tolist()]), file = fh)
                    
        return super().training_step(batch,batch_idx, optimizer_idx)

    
    def on_validation_start(self):
        return
        if hasattr(self, "cache"):
            return

        dm = self.trainer.datamodule
        num_vars = dm.y_train.shape[-1]
        self.cache = [
            torch.empty(dm.train_split, num_vars),
            torch.empty(dm.val_split, num_vars),
        ]
        for i in range(2):
            instantiate_class(self.cache[i], self.cache_init)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if not self.should_validate():
            return

        i, x, y, *meta = batch["idx"], batch["x"], batch["y"], batch["meta"]
        if i.numel() == 0:
            return []
        """
        if batch_idx == 0 and dataloader_idx == 1:
            x["visualize"] = False
            x["visualization_params"] = {
                "reduced_vis": True,
                "produce_pdf": True,
                "true_matchings": x["gt_perm_mat"],
                "string_info": "which_class",
            }
        """
        a, b, c = self.core(i, x)

        if self.hint == "none":
            h = None
        elif self.hint == "gold":
            h = y
        elif self.hint == "cache":
            h = self.cache[dataloader_idx][i]

        yhat, status = self.solver(a, b, c, h)

        if self.hint == "cache":
            self.cache[dataloader_idx][i] = yhat.detach().cpu()

        if batch_idx == 0 and dataloader_idx == 1:
            self.val_save = batch, a, b, c, yhat
            # batch, a, b, c, yhat = self.val_save

        self.validation_step_log(batch, batch_idx, dataloader_idx, a, b, c, yhat)

        return status

    def validation_step_log(self, batch, batch_idx, dataloader_idx, a, b, c, yhat):
        super().validation_step_log(batch, batch_idx, dataloader_idx, a, b, c, yhat)
        i, x, y, inputs = batch["idx"], batch["x"], batch["y"], batch["meta"]
        yhat = yhat.long()
        r = "train" if dataloader_idx == 0 else "val"
        #
        nm = int(math.sqrt(y.shape[1]))
        self.metrics[f"{r}/acc/num"](yhat.long().view(-1, nm), y.long().view(-1, nm))

        self.metrics[f"{r}/acc/pre"](
            torch.argmax(-c.contiguous().view(-1, nm), dim=-1),
            torch.argmax(y.view(-1, nm), dim=-1)
        )

        yhat = yhat.view(yhat.shape[0], nm, nm)
        is_constraint_satisfied = (
            (yhat.sum(dim=-1) == 1).all(dim=-1) & (yhat.sum(dim=-2) == 1).all(dim=-1)
        ).long()
        self.metrics[f"{r}/constraint_satisfied"](
            is_constraint_satisfied, torch.ones_like(is_constraint_satisfied).long()
        )
        """
        easy_visualize(
                orig_graph_list,
                points,
                n_points,
                images,
                unary_costs_list,
                quadratic_costs_list,
                matchings,
                **visualization_params,
                    )
        """

    def validation_epoch_end(self, outputs):
        val_metrics = super().validation_epoch_end(outputs)
        if not self.should_validate():
            return

        logger.info("acc/num ")
        logger.info(
            " \t"
            + " | ".join(
                [
                    f"train: {val_metrics['train/acc/num']:.4f}",
                    f"val: {val_metrics['val/acc/num']:.4f}",
                ]
            )
        )

    def test_step_log(self, batch, batch_idx, dataloader_idx, a, b, c, yhat):
        super().test_step_log(batch, batch_idx, dataloader_idx, a, b, c, yhat)
        i = batch["idx"]
        if batch_idx == 0:
            self.num_test_samples = i.numel()
        else:
            self.num_test_samples += i.numel()
        i, x, y, inputs = batch["idx"], batch["x"], batch["y"], batch["meta"]
        yhat = yhat.long()
        r = "test"
        nm = int(math.sqrt(y.shape[1]))
        self.metrics[f"{r}/acc/num"](yhat.long().view(-1, nm), y.long().view(-1, nm))
        self.metrics[f"{r}/acc/pre"](
            torch.argmax(-c.contiguous().view(-1, nm), dim=-1),
            torch.argmax(y.view(-1, nm), dim=-1),
        )
        yhat = yhat.view(yhat.shape[0], nm, nm)
        is_constraint_satisfied = (
            (yhat.sum(dim=-1) == 1).all(dim=-1) & (yhat.sum(dim=-2) == 1).all(dim=-1)
        ).long()
        self.metrics[f"test/constraint_satisfied"](
            is_constraint_satisfied, torch.ones_like(is_constraint_satisfied).long()
        )

    def test_epoch_end(self, outputs):
        super().test_epoch_end(outputs)

        batch, a, b, c, yhat = self.test_save
        x, y = batch["x"], batch["y"]

        logger.info("acc/num ")
        logger.info(
            " \t"
            + " | ".join(
                [
                    f"test: {self.metrics['test/acc/num'].compute().item():.4f}",
                ]
            )
        )

        logger.info("Num Test Samples: {}".format(self.num_test_samples))
