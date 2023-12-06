#!/bin/bash

cd ..

DATA_DIR="/home/ppwang/Data/NSD"

python Train_MindEye.py --data_path $DATA_DIR \
                        --model_name training \
                        --no-hidden \
                        --n_samples_save 1 \
                        --ckpt_interval 1 \
                        --no-wandb_log

cd script