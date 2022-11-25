#!/bin/bash

for n_query in 2 4 6 8 10
do
	for lr in 0.1 0.01 0.001 0.0001
	do
		LRSTR=$(echo "scale=2; $lr * 10000" | bc)
		python ./meta_train.py --data_path=/home/mrmiews/final/cats/data --dataset=cats --train_n_way=5 --val_n_way=5 --n_shot=10 --model=ResNet18 --epoch=15 --pretrain_path=/home/mrmiews/final/ResNet18_meta_deepbdc_pretrain --reduce_dim=256 --n_query=$n_query --lr=$lr --extra_dir=/n_query_${n_query}_lr_${LRSTR}/
	done
done

