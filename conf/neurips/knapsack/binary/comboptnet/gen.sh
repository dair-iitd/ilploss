#!/bin/bash
set -e

export seed items w_min w_max
for items in 10 15 20 25 30 35 40 45 50
do
	mkdir -p ${items}
	w_min=$(python -c "import math; print(math.floor(150/${items})/100)")
	w_max=$(python -c "import math; print(math.ceil(350/${items})/100)")
	for seed in {0..9}
	do
		envsubst < template.yaml > ${items}/${seed}.yaml
	done
done
