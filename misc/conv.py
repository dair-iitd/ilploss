#!/usr/bin/env python3
"""Restructure data from other projects to the current project's format.
"""

import numpy as np
import torch

import argparse
import logging
import pickle

logging.basicConfig(
    level=logging.DEBUG,
    format="\033[1m%(asctime)s\033[m %(message)s",
)


def dom_inv(args):
    logging.info(f"load {args.input}...")
    with open(args.input, "rb") as infile:
        pkl = pickle.load(infile)

    logging.info(f"len = {len(pkl)}")
    logging.info(f"convert to pt...")
    if 'target' in pkl[0].keys():
        y = torch.tensor([t["target"] for t in pkl], dtype=torch.int8)
    else:
        y = torch.tensor([t["target_set"][0] for t in pkl], dtype=torch.int8)

    pt = {
        "x": torch.tensor([t["query"] for t in pkl], dtype=torch.int8),
        "y": y, 
        "meta": {
            "box_shape": (args.n, args.m),
        },
        "__doc__": f"source {args.input}",
    }

    logging.info(f"dump {args.output}...")
    torch.save(pt, args.output)

    logging.info(f"all done.")


def comboptnet(args):
    # needs smart-settings module to load from pickle
    # install from
    # python3 -m pip install git+https://github.com/martius-lab/smart-settings.git

    logging.info(f"load pickle from {args.input}...")
    with open(args.input, "rb") as infile:
        pkl = pickle.load(infile)

    logging.info(f"convert to pytorch...")
    c, y = zip(*(pkl["train"] + pkl["test"]))
    c = torch.as_tensor(np.array(c), dtype=torch.float)
    raw_y = torch.as_tensor(np.array(y))
    if args.dense:
        y = (raw_y * 10).char()
    else:
        y = (raw_y + 0.5).char()
    meta = pkl["metadata"]
    ab = torch.as_tensor(meta["true_constraints"], dtype=torch.float)
    a, b = torch.split(ab, [ab.shape[-1] - 1, 1], dim=-1)
    a = -a
    pt = {
        "x": c,
        "y": y,
        "meta": {
            "a": a,
            "b": b,
            "c": c,
            "num_vars": meta["num_variables"],
            "num_constrs": meta["num_constraints"],
            "lb": int(meta["variable_range"]["lb"]),
            "ub": int(meta["variable_range"]["ub"]),
            "train_val_split": len(pkl["train"]),
            "test_split": len(pkl["test"]),
        },
        "__doc__": f"source {args.input}",
    }
    logging.info(f"dict to save:\n{pt}")

    logging.info(f"save to {args.output}...")
    torch.save(pt, args.output)

    logging.info("all done.")


def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers()
    parser_dom_inv = subparsers.add_parser(
        "dom_inv", help="data files from the domain invariance project"
    )
    parser_dom_inv.add_argument("input", help="input pickle file")
    parser_dom_inv.add_argument("output", help="output pytorch file")
    parser_dom_inv.add_argument("n", type=int, help="box_shape[0]")
    parser_dom_inv.add_argument("m", type=int, help="box_shape[1]")
    parser_dom_inv.set_defaults(func=dom_inv)
    parser_comboptnet = subparsers.add_parser(
        "comboptnet", help="data files from the original comboptnet repo"
    )
    parser_comboptnet.add_argument("input", help="input pickle file")
    parser_comboptnet.add_argument("output", help="output pytorch file")
    parser_comboptnet.add_argument(
        "--dense",
        action="store_true",
        help="dense dataset with domain [-5,5]; default domain is [0,1]",
    )
    parser_comboptnet.set_defaults(func=comboptnet)
    args = parser.parse_args(*args, **kwargs)
    args.func(args)


if __name__ == "__main__":
    main()
