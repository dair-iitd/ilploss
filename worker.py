#!/usr/bin/env python3

# TODO: set logger prefix/suffix hostname.pid.task
# TODO: file_handler
# TODO: fix h.npy
# inf loop
# with symlink

import gurobi as grb
import time
from gurobi import GRB
import numpy as np
from rich.logging import RichHandler
from rich.progress import Progress

import argparse
import collections
import datetime
import itertools
import json
import os
import logging
from pathlib import Path
import time
from typing import Optional
import socket

hostname = socket.gethostname()
pid = os.getpid()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = RichHandler(omit_repeated_times=False, show_path=False)
handler.setFormatter(logging.Formatter(f"{hostname}_{pid}  %(message)s"))
logger.addHandler(handler)

full_width_logger = logging.getLogger("fw" + __name__)
full_width_logger.setLevel(logging.DEBUG)
full_width_logger.addHandler(
    RichHandler(
        show_time=False, show_level=False, show_path=False, rich_tracebacks=True
    )
)

STATUS_MSG = collections.defaultdict(
    lambda: "UNKNOWN",
    {
        1: "LOADED",
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        6: "CUTOFF",
        7: "ITERATION_LIMIT",
        8: "NODE_LIMIT",
        9: "TIME_LIMIT",
        10: "SOLUTION_LIMIT",
        11: "INTERRUPTED",
        12: "NUMERIC",
        13: "SUBOPTIMAL",
        14: "INPROGRESS",
        15: "USER_OBJ_LIMIT",
    },
)


def task(w):
    a = np.load(w / "a.npy")
    b = np.load(w / "b.npy")
    c = np.load(w / "c.npy")

    with (w / "grb.log").open("a") as f:
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f" Hostname: {hostname}\n")
        f.write(f" PID: {pid}\n")
        f.write("=" * 80 + "\n")

    grb.setParam("LogFile", f"{w / 'grb.log'}")

    m = grb.Model()
    num_constrs, num_vars = a.shape
    y = m.addMVar(num_vars, lb=float("-inf"), ub=float("inf"), vtype=GRB.INTEGER)
    m.addMConstr(a, y, GRB.GREATER_EQUAL, -b)
    m.setMObjective(None, c, 0.0, sense=GRB.MINIMIZE)

    if args.mps:
        m.write(f"{w / f'{hostname}_{pid}_m.mps'}")
        return True

    if (w / "h.npy").exists():
        y.varHintVal = np.load(w / "h.npy")

    tic = time.time()
    m.optimize()
    toc = time.time()

    with (w / "info.jsonl").open("a") as f:
        json.dump({"solve_time": toc - tic, "status": STATUS_MSG[m.status]}, f)
        f.write('\n')

    logger.info(f"[{w.name}] solve_time: {datetime.timedelta(seconds=int(toc - tic))}")
    logger.info(f"[{w.name}] status: {STATUS_MSG[m.status]}")

    if m.status == GRB.OPTIMAL:
        np.save(w / f"{hostname}_{pid}_yhat.npy", y.x)
    elif m.status in {GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED}:
        np.save(w / f"{hostname}_{pid}_yhat.npy", np.full_like(c, np.nan))
    else:
        try:
            np.save(w / "h.npy", y.x)
        except grb.GurobiError:
            pass


def is_done(w):
    return len(list(w.glob("*_m.mps" if args.mps else "*_yhat.npy"))) > 0


def main():
    if args.params is not None:
        grb.readParams(args.params)

    tasks = Path(args.root_dir).iterdir()
    tasks = sorted(tasks)

    logger.info(f"{len(tasks)} tasks")

    done = set()
    last_print_time = 0
    count_complete_rounds = 0
    with Progress() as prog:
        pbar = prog.add_task(f"Tasks", total=len(tasks))
        
        for epoch in itertools.count():
            if count_complete_rounds > 10:
                logger.info("could not find any new task during last 10 epochs.. quitting ...")
                break

            this_time = time.time()
            if this_time - last_print_time > 60:
                logger.info(f"epoch: {epoch}")
            for w in tasks:
                prog.update(pbar, completed=len(done))
                if len(done) == len(tasks):
                    if this_time - last_print_time > 60:
                        logger.info("all tasks done. Taking rest and sleeping for 15 seconds")
                        last_print_time = this_time
                    count_complete_rounds += 1
                    time.sleep(15)
                    break

                if is_done(w):
                    #logger.info(f"task {w.name}: already solved.")
                    done.add(w)
                    continue

                g = list(w.glob("*_worker"))
                if g:
                    logger.info(f"[{w.name}] acquired by")
                    for p in g:
                        logger.info(f"\t{p.name}")
                    if len(g) > 1:
                        logger.warn(f"[{w.name}] {len(g)} workers on task!")
                else:
                    logger.info(f"[{w.name}] solving...")
                    lock = w / f"{hostname}_{pid}_worker"
                    lock.touch()
                    try:
                        time.sleep(1)
                        task(w)
                    except:
                        logger.exception("")
                        full_width_logger.exception("")
                    finally:
                        lock.unlink()
                    if is_done(w):
                        done.add(w)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", type=str, help="directory containing tasks")
    parser.add_argument("--params", type=str, default=None, help=".prm file for gurobi")
    parser.add_argument(
        "--mps", action="store_true", help="dump m.mps instead of solving"
    )
    args = parser.parse_args()
    main()
