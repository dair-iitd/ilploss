from . import utils

import pytorch_lightning as pl
from pytorch_lightning.accelerators import CPUAccelerator
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
from collections.abc import Sequence, Mapping
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class DictDataset(Dataset):
    def __init__(
        self,
        # data: Mapping[str, Sequence[Any]],
        data: Mapping,
    ):
        super().__init__()
        self.data = data
        assert len(data) != 0

        self.len = None
        for key, seq in data.items():
            if self.len is None:
                self.len = len(seq)
            else:
                assert self.len == len(seq)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ret = {}
        for key, seq in self.data.items():
            ret[key] = seq[idx]
        return ret


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        splits: dict,
        num_validate_train: int,
        batch_sizes: dict,
        num_workers: int = 0,
        collate_fn: Callable = default_collate,
    ):
        super().__init__()

        self.data_path = data_path
        self.splits = splits
        self.num_validate_train = num_validate_train
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        logger.info(f"load {self.data_path}...")
        pt = torch.load(self.data_path, map_location="cpu")
        logger.info(f"loaded.")

        full = {}
        full["x"] = pt["x"].float()
        full["y"] = pt["y"].float()

        assert len(full["x"]) == len(full["y"])
        assert (
            len(full["x"])
            >= self.splits["train"] + self.splits["val"] + self.splits["test"]
        )

        data = defaultdict(dict)
        for p in ["train", "val", "test"]:
            data[p]["idx"] = torch.arange(self.splits[p])

        for q in ["x", "y"]:
            data["train"][q] = full[q][: self.splits["train"]]
            data["val"][q] = full[q][
                self.splits["train"] : self.splits["train"] + self.splits["val"]
            ]
            data["test"][q] = full[q][-self.splits["test"] :]

        self.datasets = {}
        for p in ["train", "val", "test"]:
            self.datasets[p] = DictDataset(data[p])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.datasets["train"],
            batch_size=self.batch_sizes["train"],
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=self.splits["train"] % self.batch_sizes["train"] != 0,
            pin_memory=not isinstance(self.trainer.accelerator, CPUAccelerator),
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset=Subset(
                    self.datasets["train"],
                    torch.multinomial(
                        torch.ones(self.splits["train"]),
                        num_samples=self.num_validate_train,
                        replacement=False,
                    ),
                ),
                batch_size=self.batch_sizes[
                    "val" if "val" in self.batch_sizes else "test"
                ],
                num_workers=self.num_workers,
                pin_memory=not isinstance(self.trainer.accelerator, CPUAccelerator),
                collate_fn=self.collate_fn,
            ),
            DataLoader(
                dataset=self.datasets["val"],
                batch_size=self.batch_sizes[
                    "val" if "val" in self.batch_sizes else "test"
                ],
                num_workers=self.num_workers,
                pin_memory=not isinstance(self.trainer.accelerator, CPUAccelerator),
                collate_fn=self.collate_fn,
            ),
        ]

    def test_dataloader(self):
        return DataLoader(
            dataset=self.datasets["test"],
            batch_size=self.batch_sizes["test"],
            num_workers=self.num_workers,
            pin_memory=not isinstance(self.trainer.accelerator, CPUAccelerator),
            collate_fn=self.collate_fn,
        )
