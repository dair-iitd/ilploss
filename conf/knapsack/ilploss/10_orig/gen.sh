#!/bin/bash
set -e

export seed
for seed in {0..9}
do
	envsubst < template.yaml > ${seed}.yaml
    echo "timeout 12.5h python trainer.py --config conf/neurips/knapsack/binary/ilploss/10_orig/${seed}.yaml > logs/neurips/knapsack/binary/ilploss/10_orig_${seed}.log 2>&1" >> orig_10_ilploss.sh
done
