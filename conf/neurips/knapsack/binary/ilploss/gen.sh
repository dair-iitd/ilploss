#!/bin/bash
set -e

export seed items
for items in 10 15 20 25 30 35 40 45 50
do
	mkdir -p ${items}
	for seed in {0..9}
	do
		envsubst < template.yaml > ${items}/${seed}.yaml
	done
done
