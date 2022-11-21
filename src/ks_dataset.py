import pytorch_lightning as pl
import sys
from typing import Any
from .CombOptNet.src import datasets
from .utils import compute_normalized_solution, compute_denormalized_solution
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from IPython.core.debugger import Pdb

import logging

logger = logging.getLogger(__name__)


def knapsack_dataloader(dataset_path):  # , loader_params):
    variable_range = dict(lb=0, ub=1)
    num_variables = 10

    train_encodings = np.load(os.path.join(dataset_path, "train_encodings.npy"))
    train_ys = np.load(
        os.path.join(dataset_path, "train_sols.npy")
    )  # , **variable_range)
    # train_ys = compute_normalized_solution(np.load(os.path.join(dataset_path, 'train_sols.npy')), **variable_range)
    train_dataset = list(zip(train_encodings, train_ys))
    # training_set = Dataset(train_dataset)
    # train_iterator = data.DataLoader(training_set, **loader_params)

    test_encodings = np.load(os.path.join(dataset_path, "test_encodings.npy"))
    test_ys = np.load(os.path.join(dataset_path, "test_sols.npy"))
    # test_ys = compute_normalized_solution(np.load(os.path.join(dataset_path, 'test_sols.npy')), **variable_range)
    test_dataset = list(zip(test_encodings, test_ys))
    # test_set = Dataset(test_dataset)
    # test_iterator = data.DataLoader(test_set, **loader_params)

    distinct_ys_train = len(set([tuple(y) for y in train_ys]))
    distinct_ys_test = len(set([tuple(y) for y in test_ys]))
    print(
        f"Successfully loaded Knapsack dataset.\n"
        f"Number of distinct solutions in train set: {distinct_ys_train},\n"
        f"Number of distinct solutions in test set: {distinct_ys_test}"
    )

    metadata = {"variable_range": variable_range, "num_variables": num_variables}

    return (train_dataset, test_dataset), metadata


class KnapsackDataModule(datasets.BaseDataModule):
    def __init__(
        self,
        data_path: str,
        splits: dict,
        num_validate_train: int,
        batch_sizes: dict,
        num_workers: int = 0,
        eval_on_test: bool = False,
    ):

        super().__init__(
            data_path=data_path,
            splits=splits,
            num_validate_train=num_validate_train,
            batch_sizes=batch_sizes,
            num_workers=num_workers,
        )
        self.eval_on_test = eval_on_test

    def setup(self, stage=None):
        logger.info(f"load {self.data_path}...")
        (self.train_dataset, self.test_dataset), self.metadata = knapsack_dataloader(
            self.data_path
        )
        logger.info(f"loaded.")
        self.x_train, self.y_train = zip(
            *[
                [torch.from_numpy(x) for x in self.train_dataset[ind]]
                for ind in range(len(self.train_dataset))
            ]
        )
        self.train_size = len(self.x_train)
        self.x_test, self.y_test = zip(
            *[
                [torch.from_numpy(x) for x in self.test_dataset[ind]]
                for ind in range(len(self.test_dataset))
            ]
        )
        self.x_train = torch.stack(self.x_train, dim=0)
        self.y_train = torch.stack(self.y_train, dim=0).float()
        self.x_test = torch.stack(self.x_test, dim=0)
        self.y_test = torch.stack(self.y_test, dim=0).float()

        if not self.eval_on_test:
            self.x_train, self.x_val = self.split(self.x_train)
            self.y_train, self.y_val = self.split(self.y_train)
        else:
            self.x_val = self.x_test[: self.splits["val"]]
            self.y_val = self.y_test[: self.splits["val"]]

        self.x_test = self.x_test[: self.splits["test"]]
        self.y_test = self.y_test[: self.splits["test"]]

        self.datasets = {
            "train": datasets.DictDataset(
                {
                    "idx": torch.arange(self.splits["train"]),
                    "x": self.x_train,
                    "y": self.y_train,
                }
            ),
            "val": datasets.DictDataset(
                {
                    "idx": torch.arange(self.splits["val"]),
                    "x": self.x_val,
                    "y": self.y_val,
                }
            ),
            "test": datasets.DictDataset(
                {
                    "idx": torch.arange(self.splits["test"]),
                    "x": self.x_test,
                    "y": self.y_test,
                }
            ),
        }

    def split(self, x):
        total = self.train_size
        total_required = self.splits["train"] + self.splits["val"]
        ignore_split = x.shape[0] - total_required
        sz = [self.splits["train"], ignore_split, self.splits["val"]]
        train, _, val = torch.split(x, sz)
        return train, val
