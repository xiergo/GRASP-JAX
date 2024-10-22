#!/bin/bash

/lustre/grp/gyqlab/xieyh/app/anaconda3/envs/af2/bin/python /lustre/grp/gyqlab/share/xieyh/grasp_test/GRASP-JAX/run_grasp.py \
    --fasta_paths $1 \
    --data_dir /lustre/grp/gyqlab/share/AF2_database \
    --output_dir $2 \
    --feature_pickle $3 \
    --restraints_pickle $4
