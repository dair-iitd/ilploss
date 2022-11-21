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

description_str = "Script to create multinode hpc jobs. \n \
exp_{i}.sh scripts are created in args.jobs_dir. Each exp_{i}.sh script is a process to be run on one node. It fires args.num_task_per_process tasks in parallel.\n \
These processes can be run either individually - via job_{i}.sh or through one of multi_job_{k}.sh. \n \
Each multi node job multi_job_{k}.sh will run args.num_process_per_job number of processes by doing an ssh on each of the node in $PBS_NODESFILE. Ensure that passwordless ssh is enabled and number of nodes selected in args.multi_template are in sync with args.num_process_per_job. By default, each multinode jobs runs 6 processes on total of 3 nodes with 2 gpus per node. \n \
args.single_job_file submits all single jobs  job_{i}.sh via qsub. \n \
args.multi_job_file submits all multi node jobs multi_job_{k}.sh via qsub. \n\n \
Each command in exp_{i} runs args.task_script with a combination of input arguments as hard-coded in this script. Different values of an input argument should be provided as a list and a separate list for each input arg should be provided. e.g. params1, params2 and params3 in the code below.  #Tasks = Cross product of params1, params2 and params3.\n \
Jobs are sorted in the decreasing order of time it takes to run them. \n \
Time of each job is decided by one of the arguments to the task script. 'timing_key' in the code below should be set to the argument name that decides the time. 'timing' list contains the time for each job. \n \
NOTE: you may have to modify the last multi node job manually if total number of tasks to be run is not a multiple of args.num_process_per_job*args.num_task_per_process.  \n\
"


parser = argparse.ArgumentParser(description=description_str)
parser.add_argument(
    "-num_task_per_process",
    default=1,
    type=int,
    help="num tasks to run in parallel in each process",
    required=True,
)
parser.add_argument(
    "-num_process_per_job",
    default=6,
    type=int,
    help="num processes to be run in each multinode job",
    required=True,
)
# parser.add_argument('-task_script', required=True,
#                    type=str, help='path to the task script')

parser.add_argument("-template", default="single_run.sh", required=False, type=str)
parser.add_argument(
    "-multi_header", default="multinode_header.sh", required=False, type=str
)
parser.add_argument(
    "-multi_template", default="multinode_run.sh", required=False, type=str
)
parser.add_argument(
    "-single_job_file", default="all_single_jobs.sh", type=str, required=False
)
parser.add_argument(
    "-multi_job_file", default="all_multi_jobs.sh", type=str, required=False
)
parser.add_argument(
    "-jobs_dir",
    default="multinodejobs",
    type=str,
    help="directory to be created where all generated files/scripts will reside",
)
parser.add_argument("-job_name", default="mnj", type=str)
parser.add_argument("-selectos", default=" ", type=str)
parser.add_argument("-global_time", required=True, type=str)
#parser.add_argument("--dump_dir", required=True, type=str)
parser.add_argument("-test_only", required=True, type=int)
parser.add_argument("-mode", required=False, default="val", type=str)
parser.add_argument("-checkpoint_file", default=None, type=str)
parser.add_argument("-checkpoint_dir", default=None, type=str)
parser.add_argument("-config", type=str)
parser.add_argument("-split_into", type=int, default=2)
# parser.add_argument('-param_dump_dir',type=int, default=1)
parser.add_argument("-prefix", type=str, default="")
args = parser.parse_args(sys.argv[1:])

######################
# To be changed as per the input arguments of the task_script ####
# In the demo example, dummy_task_script.py takes three input arguments named input1, input2 and input3. Timing for each job has to be decided by timing_key parameter

# module_load_str = 'module load apps/pythonpackages/3.6.0/pytorch/0.4.1/gpu'
# module_load_str = 'module load apps/anaconda3/4.6.9'
# module_load_str = 'module load apps/anaconda/3'
# module_load_str = 'module load apps/pytorch/1.5.0/gpu/anaconda3'

# working_dir = '/home/cse/phd/csz178057/phd/domain-size-inv'

config = yaml.safe_load(open(args.config))

if "working_dir" in config:
    working_dir = config["working_dir"]
else:
    working_dir = os.path.dirname(os.path.join(os.getenv("PWD"), args.task_script))

ack_dir = os.path.join(os.getenv("PWD"), args.jobs_dir)

module_load_str = config["module_load_str"]

cross_product_params = config["cross_product_params"]
names = [x["name"] for x in cross_product_params]
short_names = [x["short_name"] for x in cross_product_params]

#all_params = [x["values"] for x in cross_product_params]
all_params = []
for x in cross_product_params:
    if type(x['values']) == list:
        all_params.append(x['values'])
    else:
        extr_fn = getattr(hpc_utils, x['values']['fn_name'])
        all_params.append(extr_fn(**x['values']['init_args']))


assert len(names) == len(short_names)
assert len(names) == len(all_params)

timing_key = cross_product_params[0]["name"]
timing = None
for x in cross_product_params:
    if "time_hint" in x:
        timing_key = x["name"]
        timing = x["time_hint"]
        assert len(timing) == len(x["values"])

if timing is None:
    timing = [args.global_time] * len(all_params[0])

assert len(all_params[names.index(timing_key)]) == len(
    timing
), "len of timing should be same as len of timing_key param"

timing_dict = dict(zip(all_params[names.index(timing_key)], timing))
jobs = list(itertools.product(*all_params))

additional_short_names = config["param_groups"][
    "short_names"
]  #  ['agtr','rr','sll','tll','b','fa','ans','ns','e','ne']
additional_names = config["param_groups"][
    "names"
]  # ['aug_trans','rev_reg','sym_loss_lambda','trans_loss_lambda','batch','filter_annotated','aug_num_sentences','num_sentences','epochs','num_evals']

additional_job_list = config["param_groups"]["values_list"]

if config["order"]:
    names = names + additional_names
    short_names = short_names + additional_short_names
    all_jobs = list(itertools.product(jobs, additional_job_list))
else:
    names = additional_names + names
    short_names = additional_short_names + short_names
    all_jobs = list(itertools.product(additional_job_list, jobs))

assert len(names) == len(short_names)
name_to_short = dict(zip(names, short_names))

sorted_names = copy.deepcopy(names)
sorted_names.sort()

time_header = "#PBS -l walltime={}:00"
# PBS -q workshop

if not os.path.exists(ack_dir):
    os.makedirs(ack_dir)

slurm_cmd = open(args.template).read() + "\n"
slurm_cmd = slurm_cmd.replace("${selectos}", args.selectos)

pid_closing = "for pid in ${pids[*]}; do \n \
        wait $pid \n\
done\n"

# hack_str = ". /etc/profile.d/modules.sh"
hack_str = " "
multi_header = open(args.multi_header).read()
multi_header = multi_header.replace("${selectos}", args.selectos)
multi_header = multi_header.replace("${num_nodes}", str(args.num_process_per_job // 2))

multi_run_script = open(args.multi_template).read()
multi_run_script = multi_run_script.replace("${exp_dir}", ack_dir)

base_cmd = config["base_cmd"]
if args.test_only:
    base_cmd += " --test_only 1"

# base_cmd = base_cmd.format(args.task_script)
do_not_pass_params = config["do_not_pass_params"]

jobs_dict = {}
job_name_to_time = {}

transform_fns = {}
for param_name in sorted_names:
    if param_name in config["parse_value_fns"]:
        transform_fns[param_name] = getattr(
            hpc_utils, config["parse_value_fns"][param_name]
        )
    else:
        transform_fns[param_name] = hpc_utils.replace_spl

for i, setting in enumerate(all_jobs):
    setting = list(itertools.chain(*setting))
    name_setting = {n: s for n, s in zip(names, setting)}
    # log_str = '_'.join(['%s-%s' % (name_to_short[n].replace('_', '.'),
    #                               str(name_setting[n]).replace(' ','.').replace('_','.').replace('/','.')) for n in sorted_names])

    log_str = "_".join(
        [
            "%s-%s"
            % (
                hpc_utils.replace_spl(name_to_short[n]),
                transform_fns[n](str(name_setting[n])),
            )
            for n in sorted_names
        ]
    )
    print(log_str)

    # log_str_passonly = '_'.join(['%s-%s' % (name_to_short[n].replace('_', '.'),
    #                               str(name_setting[n]).replace(' ','.').replace('_','.')) for n in sorted_names if n not in do_not_pass_params])
    setting_list = [
        "--%s %s" % (name, str(value))
        for name, value in name_setting.items()
        if name not in do_not_pass_params
    ]
    setting_str = " ".join(setting_list)

    # setting_str += ' --trainer.logger.init_args.name {}'.format(os.path.join(args.dump_dir, log_str))
    """
    if args.param_dump_dir: 
        setting_str += ' --output_dir {}'.format(os.path.join(args.dump_dir,log_str))
    else: 
        setting_str += ' --output_dir {}'.format(args.dump_dir)
    #
    """
    jobs_dict[log_str] = setting_str
    # Pdb().set_trace()
    job_name_to_time[log_str] = timing_dict[name_setting[timing_key]]

sorted_job_names = list(job_name_to_time.keys())
sorted_job_names.sort(key=lambda x: job_name_to_time[x], reverse=True)
# Pdb().set_trace()

print("Running %d jobs" % (len(jobs_dict)))

all_commands = []
hpcfile = os.path.join(args.jobs_dir, args.single_job_file)
fh = open(hpcfile, "w")
mode = stat.S_IROTH | stat.S_IRWXU | stat.S_IXOTH | stat.S_IRGRP | stat.S_IXGRP
log_str_single_job_file = os.path.join(
    args.jobs_dir, args.single_job_file + "_logstr.txt"
)
log_str_file = open(log_str_single_job_file, "w")
count = 0
jcount = 0
mjcount = 0
fhj = None
# for log_str, setting_str in jobs_dict.items():
for log_str in sorted_job_names:
    setting_str = jobs_dict[log_str]
    bash_cmd = "{} {}".format(base_cmd, setting_str)
    if count % args.num_task_per_process == 0:
        if fhj is not None:
            print(pid_closing, file=fhexp)
            print("touch {}/JACK_{}".format(ack_dir, jcount), file=fhexp)
            fhexp.close()
            print("bash {}".format(os.path.basename(tfname)), file=fhj)
            fhj.close()
            os.chmod(tfname, mode)
            os.chmod(tfname_job, mode)
            print("qsub {}".format(os.path.basename(tfname_job)), file=fh)
            jcount += 1

        if jcount % args.num_process_per_job == 0:
            print(
                "Creating new multi job. count: {},  jcount: {}, mjcount: {}".format(
                    count, jcount, mjcount
                )
            )
            fhmjname = os.path.join(args.jobs_dir, "multi_job_" + str(mjcount) + ".sh")
            fhmj = open(fhmjname, "w")
            header = "#PBS -N {}_mn_{}_{}".format(args.job_name, mjcount, log_str[:10])
            print(header, file=fhmj)
            print(time_header.format(job_name_to_time[log_str]), file=fhmj)
            print(multi_header, file=fhmj)
            print("count={}".format(jcount), file=fhmj)
            print(multi_run_script, file=fhmj)
            fhmj.close()
            os.chmod(fhmjname, mode)

            mjcount += 1

        tfname = os.path.join(args.jobs_dir, "exp_" + str(jcount) + ".sh")
        tfname_job = os.path.join(args.jobs_dir, "job_" + str(jcount) + ".sh")
        fhj = open(tfname_job, "w")
        fhexp = open(tfname, "w")
        this_time_header = time_header.format(job_name_to_time[log_str])
        header = "#PBS -N job_{}_{}\n{}\n{}\n".format(
            jcount, log_str[:10], this_time_header, slurm_cmd
        )
        print(header, file=fhj)

        print(hack_str, file=fhexp)
        print(module_load_str, file=fhexp)
        print("cd {}".format(working_dir), file=fhexp)
        print("rm {}/JACK_{}".format(ack_dir, jcount), file=fhexp)
        # print('export PATH="$(pwd)"/third_party/Jacinle/bin:$PATH', file=fhexp)

    print("count: {},  jcount: {}, mjcount: {}".format(count, jcount, mjcount))
    print("{} &".format(bash_cmd), file=fhexp)
    all_commands.append(bash_cmd)
    print("pids[{}]=$!".format(count % args.num_task_per_process), file=fhexp)
    print("{} {}".format(count, log_str), file=log_str_file)
    count += 1

if fhj is not None:
    print("Closing last job")
    print(pid_closing, file=fhexp)
    print("touch {}/JACK_{}".format(ack_dir, jcount), file=fhexp)
    fhexp.close()
    print("bash {}".format(os.path.basename(tfname)), file=fhj)
    fhj.close()
    os.chmod(tfname, mode)
    os.chmod(tfname_job, mode)
    print("qsub {}".format(os.path.basename(tfname_job)), file=fh)
    # jcount += 1
    if jcount % args.num_process_per_job == 0:
        print("Writing last multi job")
        fhmjname = os.path.join(args.jobs_dir, "multi_job_" + str(mjcount) + ".sh")
        fhmj = open(fhmjname, "w")
        header = "#PBS -N {}_mn_{}_{}\n{}\n".format(
            args.job_name, mjcount, log_str[:10], slurm_cmd
        )
        print(header, file=fhmj)
        print(multi_header, file=fhmj)
        print("count={}".format(jcount), file=fhmj)
        print(multi_run_script, file=fhmj)
        fhmj.close()
        os.chmod(fhmjname, mode)
        mjcount += 1


fh.close()
os.chmod(hpcfile, mode)
log_str_file.close()

all_multi_file_name = os.path.join(os.getenv("PWD"), args.jobs_dir, args.multi_job_file)
fh = open(all_multi_file_name, "w")
for i in range(mjcount):
    print("qsub multi_job_{}.sh".format(i), file=fh)

fh.close()
os.chmod(all_multi_file_name, mode)

# write commands to files

buckets = defaultdict(list)

for i, this_bash_cmd in enumerate(all_commands):
    bucket_id = i % args.split_into
    tmp = this_bash_cmd.split()
    tmp[0] = "python"
    tmp = " ".join(tmp)
    buckets[bucket_id].append(tmp)


for k, v in buckets.items():
    fname = os.path.join(
        os.getenv("PWD"), args.jobs_dir, args.prefix + "all_commands_" + str(k) + ".sh"
    )
    with open(fname, "w") as fh:
        print("\n".join(v), file=fh)
    #
    st = os.stat(fname)
    os.chmod(fname, st.st_mode | stat.S_IEXEC)


print("Finished")
