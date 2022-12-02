#!/bin/bash
set -e

export seed
for seed in {0..9}
do
	envsubst < template.yaml > ${seed}.yaml
    echo "python trainer.py --config conf/neurips/knapsack/binary/comboptnet/10_orig/${seed}.yaml > logs/neurips/knapsack/binary/comboptnet/10_orig_${seed}.log 2>&1" >> all_comboptnet.sh
done
