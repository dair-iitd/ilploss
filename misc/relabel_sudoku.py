#!/usr/bin/env python3
"""Relabel sudoku digits with random permutations (one for each sudoku).
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

    n, m = pt["meta"]["box_shape"]
    num_classes = n * m
    batch_size = pt["x"].shape[0]
    logging.info(f"{num_classes = }")
    logging.info(f"{batch_size = }")

    logging.info(f"make random permutations...")
    p = torch.argsort(torch.rand(batch_size, num_classes, device=device), dim=-1)

    logging.info(f"relabel query tensor...")
    q = zero_one_hot_decode(
        torch.gather(
            zero_one_hot_encode(pt["x"].to(device).long(), num_classes),
            -1,
            p[:, None, :].expand(-1, num_classes ** 2, -1),
        )
    ).cpu()

    logging.info(f"relabel target tensor...")
    t = zero_one_hot_decode(
        torch.gather(
            zero_one_hot_encode(pt["y"].to(device).long(), num_classes),
            -1,
            p[:, None, :].expand(-1, num_classes ** 2, -1),
        )
    ).cpu()

    ret = {
        "x": q.byte(),
        "y": t.byte(),
        "meta": {
            "box_shape": pt["meta"]["box_shape"],
        },
        "__doc__": pt["__doc__"] + " + relabel",
    }
    logging.info(f"output dict:\n{ret}")

    logging.info(f"save at {args.output}...")
    torch.save(ret, args.output)

    logging.info(f"done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="input pt file")
    parser.add_argument("output", help="output pt file")
    args = parser.parse_args()
    main(args)
