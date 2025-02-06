#!/bin/bash

/lustre/grp/gyqlab/xieyh/app/anaconda3/envs/af2/bin/python ./run_grasp.py \
    --data_dir /lustre/grp/gyqlab/share/AF2_database \
    --feature_pickle $1 \
    --restraints_pickle $2 \
    --output_dir $3
