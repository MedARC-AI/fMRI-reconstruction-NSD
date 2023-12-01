#!/bin/bash

cd ..

DATA_DIR="/scratch/yl6624/Data/natural-scences-dataset"

python Train_MindEye.py --data_path $DATA_DIR \
                        --model_name mindeye \
                        --no-prior

cd script