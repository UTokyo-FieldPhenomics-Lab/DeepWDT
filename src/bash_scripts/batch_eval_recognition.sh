#!/bin/bash

# Source conda environment
source /c/Users/CREST/anaconda3/etc/profile.d/conda.sh
conda activate /c/Users/CREST/anaconda3/envs/wdd

# Define arrays
v_values=(yowo_v2_nano yowo_v2_tiny)
k_values=(1 2 4 8 16 32)
e_values=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
metrics=(--cal_frame_mAP --cal_video_mAP)

# Loop over the values and execute the python script
for v in "${v_values[@]}"; do
  for k in "${k_values[@]}"; do
    for e in "${e_values[@]}"; do
      for metric in "${metrics[@]}"; do
        echo "Running eval_recognition.py with version=$v, len_clip=$k, epoch=$e, metric=$metric"
        python -m src.evaluator.eval_recognition \
          --version "$v" \
          --len_clip "$k" \
          --img_size 224 \
          "$metric" \
          --eval_split val \
          --dataset training_dataset \
          --epoch "$e"
      done
    done
  done
done