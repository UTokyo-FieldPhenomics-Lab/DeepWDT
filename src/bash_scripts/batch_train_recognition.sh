#!/bin/bash

source /c/Users/CREST/anaconda3/etc/profile.d/conda.sh
conda activate /c/Users/CREST/anaconda3/envs/deepwdt

k_values=(1 2 4 8 16 32)
v_values=(yowo_v2_nano yowo_v2_tiny)

for k_val in "${k_values[@]}"; do
    for v_val in "${v_values[@]}"; do
        echo "Running with value K: $k_val and model version: $v_val"
        python -m src.training.train_recognition --dataset training_dataset --version "$v_val" --max_epoch 20 --len_clip "$k_val"
    done
done