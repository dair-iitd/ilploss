#!/bin/bash
set -e

export seed
for seed in {0..9}
do
	envsubst < template.yaml > ${seed}.yaml
done
