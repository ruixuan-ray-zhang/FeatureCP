#!/bin/bash

# Define the list of alpha values
alphas=(0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Loop through each alpha value and run the Python script
for alpha in "${alphas[@]}"
do
    echo "Running with alpha=$alpha"
    python main_fcn.py --device 0 --data shanghaitech --dataset-dir '/home/vaecount/kevinyao/data/shanghaitech/part_B' \
    --batch_size 1 --feat_step 200 --feat_lr 0.001 --workers 10 --seed 0 --model mcnn --visualize --alpha $alpha
done
