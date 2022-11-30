#!/bin/bash

for test_n_way in 2 4 6 8
do
	for n_shot in 1 5 10
	do
		python ./test.py --data_path=/home/mrmiews/final/birds --dataset=birds --test_n_way=$test_n_way --n_shot=$n_shot --model=ResNet18 --model_path=/home/mrmiews/final/DeepBDC_ECE685/checkpoints/birds/ResNet18_meta_deepbdc_5way_10shot_metatrain/n_query_8_lr_10.000 --reduce_dim=256 --n_query=8
	done
done

