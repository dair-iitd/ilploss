#!/usr/bin/env bash

export t m i c

for t in {binary,dense}_random
do
	for m in 1 2 4 8
	do
		for i in {0..9}
		do
			c=$((2*m)) envsubst < conf/template.yaml > conf/$t/${m}x16/$i.yaml
		done
	done
done

