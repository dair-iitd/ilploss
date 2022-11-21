from __future__ import print_function
import copy
import stat
import random
from time import sleep
import os
import sys
import argparse
import itertools
import yaml
from collections import defaultdict
from IPython.core.debugger import Pdb

import hpc_utils

parser = argparse.ArgumentParser()

parser.add_argument("-config", type=str)
parser.add_argument("-dump_dir", required=True, type=str)
parser.add_argument("-output_dir", required=True, type=str)
args = parser.parse_args(sys.argv[1:])
config = yaml.safe_load(open(args.config))

os.makedirs(args.dump_dir, exist_ok=True)

cross_product_params = config["cross_product_params"]
names = [x["name"] for x in cross_product_params]
short_names = [x["short_name"] for x in cross_product_params]
all_params = [x["values"] for x in cross_product_params]
create_dirs = [x.get("create_dir", False) for x in cross_product_params]

assert len(names) == len(short_names)
assert len(names) == len(all_params)

additional_short_names = config["param_groups"][
    "short_names"
]  #  ['agtr','rr','sll','tll','b','fa','ans','ns','e','ne']
additional_names = config["param_groups"][
    "names"
]  # ['aug_trans','rev_reg','sym_loss_lambda','trans_loss_lambda','batch','filter_annotated','aug_num_sentences','num_sentences','epochs','num_evals']
additional_create_dirs = config["param_groups"]["create_dirs"]
additional_job_list = config["param_groups"]["values_list"]

jobs = list(itertools.product(*all_params))

if config["order"]:
    names = names + additional_names
    short_names = short_names + additional_short_names
    create_dirs = create_dirs + additional_create_dirs
    all_jobs = list(itertools.product(jobs, additional_job_list))
else:
    names = additional_names + names
    short_names = additional_short_names + short_names
    create_dirs = additional_create_dirs + create_dirs
    all_jobs = list(itertools.product(additional_job_list, jobs))

assert len(names) == len(short_names)
name_to_short = dict(zip(names, short_names))
name_to_create_dir = dict(zip(names, create_dirs))

sorted_names = copy.deepcopy(names)
sorted_names.sort()


config_files = config["config_files"]

transform_fns = {}
for param_name in sorted_names:
    if param_name in config["parse_value_fns"]:
        transform_fns[param_name] = getattr(
            hpc_utils, config["parse_value_fns"][param_name]
        )
    else:
        transform_fns[param_name] = hpc_utils.replace_spl


for this_config in config_files:
    for i, setting in enumerate(all_jobs):
        with open(this_config, "r") as fh:
            this_config_txt = fh.read()
            setting = list(itertools.chain(*setting))
            name_setting = {n: s for n, s in zip(names, setting)}
            log_str = "_".join(
                [
                    "%s-%s"
                    % (
                        hpc_utils.replace_spl(name_to_short[n]),
                        transform_fns[n](str(name_setting[n])),
                    )
                    for n in sorted_names
                    if name_to_short[n] != ""
                ]
            )

            sub_dirs = [
                str(transform_fns[n](str(name_setting[n])))
                for n in sorted_names
                if name_to_create_dir[n]
            ]

            for k, v in name_setting.items():
                this_config_txt = this_config_txt.replace(k, str(v))
            #
            config_base_name = os.path.basename(this_config)[:-5]
            this_config_name = config_base_name + "_" + log_str + ".yaml"
            logger_dir = os.path.join(
                args.output_dir, *sub_dirs, config_base_name, log_str
            )
            print("logger_dir:", logger_dir)
            this_config_txt = this_config_txt.replace("${logger_dir}", logger_dir)
            output_file = os.path.join(args.dump_dir, this_config_name)
            with open(output_file, "w") as wfh:
                wfh.write(this_config_txt)
