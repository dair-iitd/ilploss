#!/usr/bin/env python3
"""Generate Knapsack dataset.
"""

import sys

sys.path.insert(0, ".")
from src.CombOptNet.src.ilp import CombOptNet

import gurobi as grb
from gurobi import GRB
import numpy as np
from rich.logging import RichHandler
import torch

import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(
    RichHandler(omit_repeated_times=False, show_level=False, show_path=False)
)


def main(args):
    logger.info(f"load {args.enc_path}...")
    np_enc = torch.as_tensor(np.load(args.enc_path)).float()

    logger.info(f"split sentences...")
    ignore_sentences = np_enc.shape[0] - (
        args.train_sentences + args.val_sentences + args.test_sentences
    )
    enc = {}
    enc["train"], enc["val"], _, enc["test"] = torch.split(
        np_enc,
        [
            args.train_sentences,
            args.val_sentences,
            ignore_sentences,
            args.test_sentences,
        ],
        dim=0,
    )
    logger.info(f"{enc['train'].shape = }")
    logger.info(f"{enc['val'].shape = }")
    logger.info(f"{enc['test'].shape = }")

    logger.info(f"load {args.wp_path}...")
    np_wp = torch.as_tensor(np.load(args.wp_path)).float()

    logger.info(f"split weights and prices...")
    ignore_sentences = np_wp.shape[0] - (
        args.train_sentences + args.val_sentences + args.test_sentences
    )
    wp = {}
    wp["train"], wp["val"], _, wp["test"] = torch.split(
        np_wp,
        [
            args.train_sentences,
            args.val_sentences,
            ignore_sentences,
            args.test_sentences,
        ],
        dim=0,
    )

    logger.info(f"create random instances...")
    rand_idx = {
        "train": torch.randint(enc["train"].shape[0], [args.train_split, args.items]),
        "val": torch.randint(enc["val"].shape[0], [args.val_split, args.items]),
        "test": torch.randint(enc["test"].shape[0], [args.test_split, args.items]),
    }
    # logger.info(f"{rand_idx = }")
    logger.info(f"{rand_idx['train'].shape = }")
    logger.info(f"{rand_idx['val'].shape = }")
    logger.info(f"{rand_idx['test'].shape = }")

    if args.capacity is None:
        logger.warning(f"capacity was not given")
        logger.warning(f"using capacity = 10 x items")
        args.capacity = float(10 * args.items)
    logger.info(f"capacity = {args.capacity}")

    logger.info(
        f"instantiating solver with {args.workers} workers\n"
        f"\tand {args.threads} threads per worker..."
    )
    solver = CombOptNet(
        vtype="B",
        env=grb.Env(params={"OutputFlag": 0, "Threads": args.threads}),
        num_workers=args.workers,
        show_tqdm=True,
    )

    logger.info(f"make ILP instances...")
    y = {}
    a = {}
    b = {}
    c = {}
    for p in ["train", "val", "test"]:
        logger.info(f"___ {p = } ___")
        a[p] = -wp[p][:, 0][rand_idx[p]][:, None, :]
        b[p] = torch.empty(a[p].shape[0], 1).fill_(args.capacity)
        c[p] = -wp[p][:, 1][rand_idx[p]]
        # logger.info(f"{a[p] = }")
        # logger.info(f"{b[p] = }")
        # logger.info(f"{c[p] = }")
        logger.info(f"{a[p].shape = }")
        logger.info(f"{b[p].shape = }")
        logger.info(f"{c[p].shape = }")

        logger.info(f"solving {a[p].shape[0]} instances...")
        y[p], status = solver(a[p], b[p], c[p], None)
        assert torch.all(status == GRB.OPTIMAL)
        # logger.info(f"{y[p] = }")

    logger.info(f"constructing pt object...")
    pt = {
        "encodings": enc,
        "capacity": args.capacity,
        "rand_idx": rand_idx,
        "y": y,
        "meta": {
            "weights_prices": wp,
            "a": a,
            "b": b,
            "c": c,
        },
        "__doc__": f"source {args.enc_path} + {args.wp_path}",
    }
    # logger.info(f"{pt = }")

    if args.out_path is None:
        args.out_path = f"data/knapsack/{args.items}.pt"
    logger.info(f"dumping at {args.out_path}")
    torch.save(pt, args.out_path)
    logger.info(f"done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("items", type=int, help="number of items in each instance")
    parser.add_argument(
        "--out_path",
        default=None,
        help="output pt file (default 'data/knapsack/<items>.pt')",
    )
    parser.add_argument(
        "--enc_path",
        default="data/knapsack/orig/sentences.npy",
        help="path to `sentences.npy' file (default 'data/knapsack/orig/sentences.npy')",
    )
    parser.add_argument(
        "--wp_path",
        default="data/knapsack/orig/weights_prices.npy",
        help="path to `weights_prices.npy' file (default 'data/knapsack/orig/weights_prices.npy')",
    )
    parser.add_argument(
        "--train_sentences",
        type=int,
        default=40000,
        help="number of sentences to use for training (default 40_000)",
    )
    parser.add_argument(
        "--val_sentences",
        type=int,
        default=5000,
        help="number of sentences to use for validation (default 5_000)",
    )
    parser.add_argument(
        "--test_sentences",
        type=int,
        default=5000,
        help="number of sentences to use for testing (default 5_000)",
    )
    parser.add_argument(
        "--train_split",
        type=int,
        # default=4000,
        # help="number of knapsack instances for training (default 4_000)",
        default=10_000,
        help="number of knapsack instances for training (default 10_000)",
    )
    parser.add_argument(
        "--val_split",
        type=int,
        # default=500,
        # help="number of knapsack instances for validation (default 500)",
        default=1_000,
        help="number of knapsack instances for validation (default 1_000)",
    )
    parser.add_argument(
        "--test_split",
        type=int,
        # default=500,
        # help="number of knapsack instances for testing (default 500)",
        default=1_000,
        help="number of knapsack instances for testing (default 1_000)",
    )
    parser.add_argument(
        "--capacity",
        type=float,
        default=None,
        help="capacity of knapsack (default 10 * <items>)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of parallel workers for Gurobi (default 1)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="number of threads per worker Gurobi (default 1)",
    )
    args = parser.parse_args()
    main(args)
