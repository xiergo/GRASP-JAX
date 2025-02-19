#!/bin/bash

fasta_path=$1
restraints_file=$2
output_dir=$3
rank_by=${4:-"plddt"} # plddt or ptm, default is plddt
reduce_db=${5:-0} # 0 or 1, 0 means using full database (default), 1 means using reduced database

if [ $reduce_db -eq 0 ]; then

    echo "Searching for MSA using full database"
    /lustre/grp/gyqlab/xieyh/app/anaconda3/envs/GRASP/bin/python ./run_grasp.py \
        --data_dir /lustre/grp/gyqlab/share/AF2_database \
        --jackhmmer_binary_path /lustre/grp/gyqlab/xieyh/app/hmmer-3.4/hmmer-3.4/bin/jackhmmer \
        --hhblits_binary_path /lustre/grp/gyqlab/xieyh/app/hh-suite/build/bin/hhblits \
        --hhsearch_binary_path /lustre/grp/gyqlab/xieyh/app/hh-suite/build/bin/hhsearch \
        --hmmsearch_binary_path /lustre/grp/gyqlab/xieyh/app/hmmer-3.4/hmmer-3.4/bin/hmmsearch \
        --hmmbuild_binary_path /lustre/grp/gyqlab/xieyh/app/hmmer-3.4/hmmer-3.4/bin/hmmbuild \
        --kalign_binary_path /lustre/grp/gyqlab/xieyh/app/kalign-3.3.5/build/bin/kalign \
        --uniref90_database_path=/lustre/grp/gyqlab/share/AF2_database/uniref90/uniref90.fasta \
        --mgnify_database_path=/lustre/grp/gyqlab/share/AF2_database/mgnify/mgy_clusters.fa \
        --template_mmcif_dir=/lustre/grp/gyqlab/share/AF2_database/pdb_mmcif/mmcif_files \
        --obsolete_pdbs_path=/lustre/grp/gyqlab/share/AF2_database/pdb_mmcif/obsolete.dat \
        --uniprot_database_path=/lustre/grp/gyqlab/share/AF2_database/uniprot/uniprot.fasta \
        --pdb_seqres_database_path=/lustre/grp/gyqlab/share/AF2_database/pdb_seqres/pdb_seqres.txt \
        --uniref30_database_path=/lustre/grp/gyqlab/share/AF2_database/uniref30/UniRef30_2021_03 \
        --bfd_database_path=/lustre/grp/gyqlab/share/AF2_database/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
        --max_template_date=2021-10-08 \
        --db_preset=full_dbs \
        --use_precomputed_msas=true \
        --logtostderr \
        --random_seed 0 \
        --use_gpu_relax=true \
        --num_multimer_predictions_per_model=5 \
        --models_to_relax=best \
        --rank_by $rank_by \
        --fasta_path $fasta_path \
        --restraints_file $restraints_file \
        --output_dir $output_dir
else
    echo "Searching for MSA using reduced database"
    /lustre/grp/gyqlab/xieyh/app/anaconda3/envs/GRASP/bin/python ./run_grasp.py \
        --data_dir /lustre/grp/gyqlab/share/AF2_database \
        --jackhmmer_binary_path /lustre/grp/gyqlab/xieyh/app/hmmer-3.4/hmmer-3.4/bin/jackhmmer \
        --hhblits_binary_path /lustre/grp/gyqlab/xieyh/app/hh-suite/build/bin/hhblits \
        --hhsearch_binary_path /lustre/grp/gyqlab/xieyh/app/hh-suite/build/bin/hhsearch \
        --hmmsearch_binary_path /lustre/grp/gyqlab/xieyh/app/hmmer-3.4/hmmer-3.4/bin/hmmsearch \
        --hmmbuild_binary_path /lustre/grp/gyqlab/xieyh/app/hmmer-3.4/hmmer-3.4/bin/hmmbuild \
        --kalign_binary_path /lustre/grp/gyqlab/xieyh/app/kalign-3.3.5/build/bin/kalign \
        --uniref90_database_path=/lustre/grp/gyqlab/share/AF2_database/uniref90/uniref90.fasta \
        --mgnify_database_path=/lustre/grp/gyqlab/share/AF2_database/mgnify/mgy_clusters.fa \
        --template_mmcif_dir=/lustre/grp/gyqlab/share/AF2_database/pdb_mmcif/mmcif_files \
        --obsolete_pdbs_path=/lustre/grp/gyqlab/share/AF2_database/pdb_mmcif/obsolete.dat \
        --uniprot_database_path=/lustre/grp/gyqlab/share/AF2_database/uniprot/uniprot.fasta \
        --pdb_seqres_database_path=/lustre/grp/gyqlab/share/AF2_database/pdb_seqres/pdb_seqres.txt \
        --small_bfd_database_path=/lustre/grp/gyqlab/share/AF2_database/small_bfd/bfd-first_non_consensus_sequences.fasta \
        --max_template_date=2021-10-08 \
        --db_preset=reduced_dbs \
        --use_precomputed_msas=true \
        --logtostderr \
        --random_seed 0 \
        --use_gpu_relax=true \
        --num_multimer_predictions_per_model=5 \
        --models_to_relax=best \
        --rank_by $rank_by \
        --fasta_path $fasta_path \
        --restraints_file $restraints_file \
        --output_dir $output_dir
fi




    
