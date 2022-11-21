#!/bin/bash
set -e

for seed in {0..9}
do
	echo
	mkdir -p "logs/${1}"
	date
	set -x
	./trainer.py --config "conf/${1}/${seed}.yaml" &>> "logs/${1}/${seed}.log"
	set +x
	echo
done
