#!/usr/bin/env python3
"""Shuffle examples in dataset.
"""

import torch
import torch.nn.functional as F

import argparse
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="\033[1m%(asctime)s\033[m %(message)s",
)


def zero_one_hot_encode(x, num_classes):
    nz = x != 0
    ret = F.one_hot(nz * (x - 1).long(), num_classes=num_classes)
    return ret * nz[..., None]


def zero_one_hot_decode(x):
    return (torch.argmax(x, dim=-1) + 1) * torch.any(x != 0, dim=-1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"{device = }")

    logging.info(f"load pt from {args.input}...")
    pt = torch.load(args.input, map_location="cpu")

    logging.info(f"input dict:\n{pt}")

    logging.info(f"make random permutation...")
    perm = torch.randperm(pt["x"].shape[0])

    logging.info(f"shuffle x...")
    pt["x"] = pt["x"][perm]

    logging.info(f"shuffle y...")
    pt["y"] = pt["y"][perm]

    logging.info(f"modify __doc__...")
    pt["__doc__"] = pt["__doc__"] + " + shuffle"

    logging.info(f"output dict:\n{pt}")

    logging.info(f"save at {args.output}...")
    torch.save(pt, args.output)

    logging.info(f"done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="input pt file")
    parser.add_argument("output", help="output pt file")
    args = parser.parse_args()
    main(args)
