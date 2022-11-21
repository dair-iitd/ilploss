#!/usr/bin/env python3
import os
from src import handler
from src import models

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
#from pytorch_lightning.cli import LightningCLI
import torch
from torch import optim

import logging
import sys
from typing import Union
from IPython.core.debugger import Pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--ckpt_path",
            type=Union[None, str],
            default=None,
            help="Resume training from *.ckpt file",
        )
        parser.add_argument(
            "--model_path",
            type=Union[None, str],
            default=None,
            help="Load model state dict from *.ckpt file",
        )
        parser.add_argument(
            "--skip_initial_validation",
            type=int,
            default=1,
            help="Skip the initial validation step",
        )
        parser.add_argument(
            "--start_val_after_n_epochs",
            type=int,
            default=0,
            help="Start validation after n epochs",
        )
        parser.add_argument(
            "--test_only",
            type=int,
            default=0,
            help="Evaluate only on test data after loading from model_path",
        )
        parser.add_argument(
            "--skip_solve",
            type=int,
            default=0,
            help="Skip solve in testing, only dump ILP instances",
        )
        parser.add_argument(
            "--log_idx",
            type=int,
            default=0,
            help="Log batch id and data id information?",
        )



if __name__ == "__main__":
    cli = MyLightningCLI(
        model_class=pl.LightningModule,
        subclass_mode_model=True,
        datamodule_class=pl.LightningDataModule,
        subclass_mode_data=True,
        run=False,
        save_config_overwrite=True,
    )
    cli.model.config = cli.config
    if cli.config["model_path"] is not None:
        logger.info(f"load model state dict from {cli.config['model_path']}")
        incompatible_keys = cli.model.load_state_dict(
            torch.load(cli.config["model_path"], map_location="cpu")["state_dict"],
        )
        if incompatible_keys:
            logger.warning(f"{incompatible_keys = }")

        if cli.config["test_only"] > 0:
            cli.trainer.test(model=cli.model, datamodule=cli.datamodule)
            sys.exit()

    if not cli.config["skip_initial_validation"]:
        cli.trainer.validate(
            cli.model, cli.datamodule, ckpt_path=cli.config["ckpt_path"]
        )
    os.makedirs(os.path.join(cli.trainer.logger.log_dir,'checkpoints'), exist_ok = True)
    torch.save(cli.model.state_dict(),os.path.join(cli.trainer.logger.log_dir,'checkpoints','start.pt'))

    #torch.use_deterministic_algorithms(True, warn_only=True)
    cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=cli.config["ckpt_path"])
    cli.trainer.test(datamodule=cli.datamodule)
