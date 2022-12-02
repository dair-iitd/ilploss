#!/usr/bin/env bash
export seed items
for items in 15 20 25 30
do
	mkdir -p ${items}
	for seed in {0..9}
	do
		envsubst < template.yaml > ${items}/${seed}.yaml
        echo "timeout 12.5h python trainer.py --config conf/neurips/knapsack/binary/ilploss/${items}/${seed}.yaml > logs/neurips/knapsack/binary/ilploss/${items}_${seed}.log 2>&1" >> all_ilploss.sh
	done
done
