from . import ilp, utils
import os
import gurobi as grb
from gurobi import GRB
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import instantiate_class
from pytorch_lightning.utilities.model_summary import summarize
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as tm

import collections
import logging
from pathlib import Path
import re
from typing import Optional
from IPython.core.debugger import Pdb

logger = logging.getLogger(__name__)


class MyLightningModule(pl.LightningModule):
    def __init__(
        self,
        optimizer: dict,
        lr_scheduler: dict,
        schedulers: dict = {},
    ):
        super().__init__()

        torch.backends.cudnn.benchmark = True

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.schedulers = schedulers

        self.metrics = nn.ModuleDict({})
        self.my_hparams = {
            "optimizer": utils.hparams(optimizer),
            "lr_scheduler": utils.hparams(lr_scheduler),
            "schedulers": utils.hparams(schedulers),
        }
        self.hparam_metrics = []

    def should_validate(self):
        start_validation_after = self.config.get("start_val_after_n_epochs", 0)
        if self.current_epoch < start_validation_after:
            return False
        return True

    def on_fit_start(self):
        hparams = utils.flatten_dict(self.my_hparams)
        hparam_str = (
            f"Hyperparameters:\n"
            f"{'-'*80}\n{utils.pretty_dict(hparams, mid=' = ')}\n{'-'*80}"
        )
        logger.info(hparam_str)
        try:
            self.logger.log_hyperparams(
                params=hparams, metrics=dict.fromkeys(self.hparam_metrics, float("nan"))
            )
        except (TypeError, AttributeError):
            pass

        try:
            tb = self.logger.experiment
            tb.add_text(
                "model_summary", utils.wrap_text(summarize(self, max_depth=-1)), 0
            )
            tb.add_text("hparams", utils.wrap_text(hparam_str))
        except AttributeError:
            pass

        for module in list(self.modules()):
            assert not hasattr(module, "model")
            module.model = lambda: self

    def training_step_end(self, loss):
        if loss is not None:
            self.log("train/loss", loss)
        g = [p.grad.view(-1) for p in self.parameters() if p.grad is not None]
        if g:
            self.log("train/grad", torch.linalg.vector_norm(torch.cat(g)))

    def configure_optimizers(self):
        custom = self.optimizer.pop("custom", {})

        all_names = set()
        reg_names = set()
        params = collections.defaultdict(list)
        for name, param in self.named_parameters():
            if param.requires_grad:
                all_names.add(name)
                for regex in custom.keys():
                    if re.fullmatch(regex, name):
                        reg_names.add(name)
                        params[regex].append(param)

        for regex in custom.keys():
            if regex not in params.keys():
                logger.warning(f"regex '{regex}' does not match any param!")

        groups = [
            {"params": params[regex], **kwargs} for regex, kwargs in custom.items()
        ]

        def_names = all_names - reg_names
        def_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name in def_names:
                    def_params.append(param)

        groups.append({"params": def_params})

        optimizer = instantiate_class(groups, self.optimizer)

        config = self.lr_scheduler.pop("config", {})
        lr_scheduler = {
            "scheduler": instantiate_class(optimizer, self.lr_scheduler),
            **config,
        }

        schedulers = []
        self.scheduler_name_to_id = {}
        dummy_optimizers = []
        for i, (name, scheduler) in enumerate(self.schedulers.items()):
            self.scheduler_name_to_id[name] = i
            init = scheduler.pop("init", 1.0)
            dummy_optimizers.append(optim.SGD([nn.Parameter(torch.zeros(1))], lr=init))
            config = scheduler.pop("config", {})
            schedulers.append(
                {
                    "scheduler": instantiate_class(dummy_optimizers[-1], scheduler),
                    **config,
                },
            )

        return [optimizer, *dummy_optimizers], [lr_scheduler, *schedulers]

    def query_scheduler(self, name):
        scheduler = self.lr_schedulers()[self.scheduler_name_to_id[name] + 1]
        return scheduler.optimizer.param_groups[0]["lr"]


class DecoderFreeModel(MyLightningModule):
    def __init__(
        self,
        x_key: str,
        y_key: str,
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
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            schedulers=schedulers,
        )

        self.x_key = x_key
        self.y_key = y_key
        self.core = core
        self.solver = solver
        self.hint = hint
        self.cache_init = cache_init
        self.reset_on_temp_change = reset_on_temp_change
        self.old_temp = None

        self.metrics.update(
            {
                "train/acc/idv": tm.Accuracy(
                    subset_accuracy=False,
                    mdmc_average="samplewise",
                ),
                "train/acc/all": tm.Accuracy(
                    subset_accuracy=True,
                    mdmc_average="samplewise",
                ),
                "val/acc/idv": tm.Accuracy(
                    subset_accuracy=False,
                    mdmc_average="samplewise",
                ),
                "val/acc/all": tm.Accuracy(
                    subset_accuracy=True,
                    mdmc_average="samplewise",
                ),
                "test/acc/idv": tm.Accuracy(
                    subset_accuracy=False,
                    mdmc_average="samplewise",
                ),
                "test/acc/all": tm.Accuracy(
                    subset_accuracy=True,
                    mdmc_average="samplewise",
                ),
            }
        )
        self.my_hparams.update(
            {
                "core": utils.hparams(self.core),
                "reset_on_temp_change": utils.hparams(reset_on_temp_change),
            }
        )
        self.hparam_metrics += [
            "val/acc/idv",
            "val/acc/all",
            "test/acc/idv",
            "test/acc/all",
        ]
        self.latest_metric_values = {}

    def on_validation_start(self):
        if self.hint == "cache" and not hasattr(self, "cache"):
            dm = self.trainer.datamodule
            batch = next(dm.train_data_loader())
            self.num_vars = batch[self.y_key].shape[-1]
            self.cache = [
                torch.empty(dm.splits["train"], self.num_vars),
                torch.empty(dm.splits["val"], self.num_vars),
            ]
            for i in range(2):
                instantiate_class(self.cache[i], self.cache_init)

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

        yhat, status = self.solver(a, b, c, h)

        if self.hint == "cache":
            self.cache[dataloader_idx][i] = yhat.detach().cpu()

        if batch_idx == 0 and dataloader_idx == 1:
            self.val_save = batch, a, b, c, yhat

        self.validation_step_log(batch, batch_idx, dataloader_idx, a, b, c, yhat)

        return status

    def validation_step_log(self, batch, batch_idx, dataloader_idx, a, b, c, yhat):
        y = batch[self.y_key]

        r = "train" if dataloader_idx == 0 else "val"
        lb = torch.minimum(torch.min(yhat).long(), batch[self.y_key].long())
        self.metrics[f"{r}/acc/idv"](yhat.long() - lb, batch[self.y_key].long() - lb)
        self.metrics[f"{r}/acc/all"](yhat.long() - lb, batch[self.y_key].long() - lb)
        if self.config['log_idx']:
            with open(os.path.join(self.logger.log_dir,'val_batches.csv'), 'a') as fh:
                idx_stats = [self.current_epoch, dataloader_idx,batch_idx] + batch['idx'].cpu().numpy().tolist()
                print(','.join(map(str,idx_stats)),file = fh)



    def reset_test_metrics(self):
        for k, v in self.metrics.items():
            if k.startswith("test"):
                v.reset()

    def reset_val_metrics(self):
        for k, v in self.metrics.items():
            if not k.startswith("test"):
                v.reset()

    def validation_epoch_end(self, outputs):
        if not self.should_validate():
            logger.info(f"skip validation for {self.current_epoch}")
            # update all the metrics so that model checkpointing doesnot fail
            # for k,v in self.metrics.items():
            #    v.update(torch.zeros(2,2).long(), torch.ones(2,2).long())
            val_metric_values = {
                k: 0.0 for k, v in self.metrics.items() if not k.startswith("test")
            }
            self.log_dict(val_metric_values)
            self.reset_val_metrics()
            self.update_latest_metric_values(val_metric_values)
            self.update_latest_metric_values({"val_update_epoch": self.current_epoch})
            # self.latest_metric_values.update(val_metric_values)
            # self.latest_metric_values['val_update_epoch'] = self.current_epoch
            return val_metric_values

        batch, a, b, c, yhat = self.val_save
        x, y = batch[self.x_key], batch[self.y_key]

        norm = torch.linalg.vector_norm(a, dim=-1)
        a = a / norm[:, :, None]
        b = b / norm
        a_t = a.transpose(-1, -2)
        b_t = b[:, None, :]

        try:
            tb = self.logger.experiment
            tb.add_histogram("a_hist", a, self.global_step)
            tb.add_histogram("b_hist", b, self.global_step)
            tb.add_figure(
                "a_fig",
                sns.heatmap(
                    a[0].detach().cpu(),
                    cmap="icefire_r",
                    center=0,
                ).get_figure(),
                self.global_step,
            )
            tb.add_figure(
                "b_fig",
                sns.heatmap(
                    b[0, :, None].detach().cpu(),
                    cmap="icefire_r",
                    center=0,
                ).get_figure(),
                self.global_step,
            )
        except AttributeError:
            pass

        sample_id = torch.randint(0, y.shape[0], (1,))[0]
        # tb.add_text(
        #     "val/sample",
        #     utils.wrap_text(
        #         f'yhat: {" ".join(map(str, yhat[sample_id].long().tolist()))}\n'
        #         f'y   : {" ".join(map(str, y[sample_id].long().tolist()))}'
        #     ),
        #     self.global_step,
        # )

        train_statuses = {
            ilp.STATUS_MSG[k]: v
            for k, v in collections.Counter(torch.cat(outputs[0]).tolist()).items()
        }
        val_statuses = {
            ilp.STATUS_MSG[k]: v
            for k, v in collections.Counter(torch.cat(outputs[1]).tolist()).items()
        }

        val_metric_values = {
            k: v.compute() for k, v in self.metrics.items() if not k.startswith("test")
        }
        self.log_dict(val_metric_values)
        self.log_dict(
            {("train/status/" + k): float(v) for k, v in train_statuses.items()}
        )
        self.log_dict({("val/status/" + k): float(v) for k, v in val_statuses.items()})

        logger.info(f"")
        logger.info(f"EPOCH {self.current_epoch}")
        logger.info("train")
        logger.info(
            " \t"
            + " | ".join(
                [
                    f"idv: {self.metrics['train/acc/idv'].compute().item():.4f}",
                    f"all: {self.metrics['train/acc/all'].compute().item():.4f}",
                ]
            )
        )
        logger.info(
            " \t" + " | ".join([f"{k}: {v}" for k, v in train_statuses.items()])
        )
        logger.info("val")
        logger.info(
            " \t"
            + " | ".join(
                [
                    f"idv: {self.metrics['val/acc/idv'].compute().item():.4f}",
                    f"all: {self.metrics['val/acc/all'].compute().item():.4f}",
                ]
            )
        )
        logger.info(" \t" + " | ".join([f"{k}: {v}" for k, v in val_statuses.items()]))
        self.reset_val_metrics()
        self.update_latest_metric_values(val_metric_values)
        self.update_latest_metric_values({"val_update_epoch": self.current_epoch})
        self.update_latest_metric_values(val_statuses, "val/status/")
        self.update_latest_metric_values(train_statuses, "train/status/")
        return val_metric_values

    def update_latest_metric_values(
        self, key_value_pairs, key_prefix="", key_suffix="", filter_fn=None
    ):
        self.latest_metric_values.update(
            {
                key_prefix + k + key_suffix: (v.item() if torch.is_tensor(v) else v)
                for k, v in key_value_pairs.items()
                if (filter_fn is None or filter_fn(k, v))
            }
        )

    def on_train_epoch_start(self):
        if not self.reset_on_temp_change:
            return
        temp = self.query_scheduler("temp")
        if (
            self.reset_on_temp_change
            and self.old_temp is not None
            and temp != self.old_temp
        ):
            self.load_state_dict(
                torch.load(Path(self.logger.log_dir) / "checkpoints" / "best.ckpt")[
                    "state_dict"
                ]
            )
            logger.info("temperature change detected!")
            logger.info("model restored to current best checkpoint")
        self.old_temp = temp

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if optimizer_idx != 0:
            return None
        i, x, y = batch["idx"], batch[self.x_key], batch[self.y_key]
        loss = self.core(i, x, y)
        if self.config['log_idx']:
            with open(os.path.join(self.logger.log_dir,'train_batches.csv'), 'a') as fh:
                idx_stats = [self.current_epoch, batch_idx] + i.cpu().numpy().tolist()
                print(','.join(map(str,idx_stats)), file = fh)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        i, x, y = batch["idx"], batch[self.x_key], batch[self.y_key]
        if i.numel() == 0:
            return torch.tensor([]).cuda()
        a, b, c = self.core(i, x)

        #Pdb().set_trace()
        if self.config["skip_solve"]:
            r = Path(self.trainer.logger.log_dir) / "pred"
            r.mkdir(exist_ok=True)
            for i_, a_, b_, c_, y_ in zip(
                i.cpu().numpy(),
                a.cpu().numpy(),
                b.cpu().numpy(),
                c.cpu().numpy(),
                y.cpu().numpy(),
            ):
                w = r / f"{i_}"
                w.mkdir()

                np.save(w / "a.npy", a_)
                np.save(w / "b.npy", b_)
                np.save(w / "c.npy", c_)
                np.save(w / "y.npy", y_)

            return

        yhat, status = self.solver(a, b, c, None)

        if batch_idx == 0 and (dataloader_idx is None or dataloader_idx == 0):
            self.test_save = batch, a, b, c, yhat

        self.test_step_log(batch, batch_idx, dataloader_idx, a, b, c, yhat)

        cur_test_statuses = {
            ilp.STATUS_MSG[k]: v
            for k, v in collections.Counter(status.tolist()).items()
        }
        try:
            self.statuses += status.tolist()
        except AttributeError:
            self.statuses = status.tolist()
        net_test_statuses = {
            ilp.STATUS_MSG[k]: v for k, v in collections.Counter(self.statuses).items()
        }

        logger.info(
            "cur_status = "
            + " | ".join([f"{k}: {v}" for k, v in cur_test_statuses.items()])
            + "\t"
            + "net_status = "
            + " | ".join([f"{k}: {v}" for k, v in net_test_statuses.items()])
        )
        logger.info(f"")

        return status

    def test_step_log(self, batch, batch_idx, dataloader_idx, a, b, c, yhat):
        y = batch[self.y_key]

        lb = torch.minimum(torch.min(yhat).long(), batch[self.y_key].long())
        self.metrics["test/acc/idv"](yhat.long() - lb, batch[self.y_key].long() - lb)
        self.metrics["test/acc/all"](yhat.long() - lb, batch[self.y_key].long() - lb)

    def test_epoch_end(self, outputs):

        logger.info(
            "find test ILP instances in "
            f"{Path(self.trainer.logger.log_dir) / 'pred'}"
        )
        if self.config["skip_solve"]:
            logger.info("skipped solve, no test metrics")
            exit()

        batch, a, b, c, yhat = self.test_save
        x, y = batch[self.x_key], batch[self.y_key]

        try:
            tb = self.logger.experiment
            tb.add_text(
                "test/sample",
                utils.wrap_text(
                    f"yhat: {' '.join(map(str, yhat[0].long().tolist()))}\n"
                    f"y   : {' '.join(map(str, y[0].long().tolist()))}"
                ),
                self.global_step,
            )
        except AttributeError:
            pass

        test_statuses = {
            ilp.STATUS_MSG[k]: v
            for k, v in collections.Counter(torch.cat(outputs).tolist()).items()
        }

        self.update_latest_metric_values(test_statuses, "test/status/")

        test_metrics = {
            k: v.compute().item()
            for k, v in self.metrics.items()
            if k.startswith("test")
        }
        self.reset_test_metrics()

        self.update_latest_metric_values(test_metrics)
        self.update_latest_metric_values({"test_update_epoch": self.current_epoch})

        self.log_dict(test_metrics)

        self.log_dict(
            {("test/status/" + k): float(v) for k, v in test_statuses.items()}
        )

        logger.info("test")
        logger.info(
            " \t"
            + " | ".join(
                [
                    f"idv: {test_metrics['test/acc/idv']:.4f}",
                    f"all: {test_metrics['test/acc/all']:.4f}",
                ]
            )
        )
        logger.info(" \t" + " | ".join([f"{k}: {v}" for k, v in test_statuses.items()]))
        return test_metrics
