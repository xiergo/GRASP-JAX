#!/bin/bash

python ./run_grasp.py \
    --rank_by plddt \
    --data_dir /lustre/grp/gyqlab/share/AF2_database \
    --feature_pickle $1 \
    --restraints_file $2 \
    --output_dir $3 \
    --fasta_path ${4:-None}
    
