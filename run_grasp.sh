#!/bin/bash

/lustre/grp/gyqlab/xieyh/app/anaconda3/envs/af2/bin/python /lustre/grp/gyqlab/share/xieyh/grasp_test/grasp_test/GRASP-JAX/run_grasp.py \
    --rank_by plddt \
    --data_dir /lustre/grp/gyqlab/share/AF2_database \
    --feature_pickle $1 \
    --restraints_file $2 \
    --output_dir $3 \
    --fasta_path ${4:-None}
    
