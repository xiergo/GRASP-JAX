![header](imgs/header.png)

# GRASP-JAX

Our tool provides accurate protein complex structure prediction, seamlessly integrated with diverse experimental restraints, using JAX. 
The original model was trained with the MindSpore version, which is available in [GRASP-MS](https://github.com/xiergo/GRASP-MS).

## Installation

To install GRASP-JAX, follow these steps:

1. Clone the repository and navigate into it.

   ```bash
   git clone https://github.com/xiergo/GRASP_JAX.git
   cd ./GRASP_JAX
   ```

2. Download the necessary genetic database as described in [AlphaFold](https://github.com/google-deepmind/alphafold). Additionally, download the GRASP model weights from [this link](https://osf.io/6kjuq/) and move them to the directory where you stored the AlphaFold genetic database:


    ```bash
    mkdir PATH_TO_ALPHAFOLD_DATASET/params/
    mv params_model_1_v3_v11_*.npz PATH_TO_ALPHAFOLD_DATASET/params/
    ```

3. create a conda Enviroment, and activate it:

   ```bash
   conda create -n GRASP python=3.8 -f requiremts.txt
   conda activate GRASP
   ```
  The installation takes ~10 minutes.
  
## Prepare restraints file

We use a dictionary to store restraint information for GRASP inference, which consists of three key components: `sbr`, `sbr_mask`, and `interface_mask`. The `interface_mask` is an array of shape (`N_residues`,), where `1` marks interface residues and `0` indicates the absence of relevant information. Here, `N_residues` stands for the total number of residues in the protein complex. The `sbr` is an array of shape (`N_residues`, `N_residues`, `N_bins`), where `N_bins` represents the number of distance bins used for the restraints, and it holds the distogram of the restraints between residue pairs. Finally, the `sbr_mask` is an array of shape (`N_residues`, `N_residues`), where `1` signifies a restraint between a pair of residues and `0` means no restraint exists.

To prepare restraints file, you can use the script `generate_restr.py`. The script will generate a pkl file for GRASP inference. To use this script, just type 'python generate_restr.py -h' to see the help message:
```bash
usage: generate_restr.py [-h] [--output_file OUTPUT_FILE] restraints_file fasta_file

positional arguments:
  restraints_file       path to the restraints file. Each line contains one restraint, with the format: "chainindex1-residueindex1-residuetype1, chainindex1-residueindex1-residuetype1, distance_cutoff[, fdr]" for the RPR restraint, "chainindex1-residueindex1-residuetype1" for the IR restraint.
  fasta_file            path to the fasta file

optional arguments:
  -h, --help            show this help message and exit
  --output_file OUTPUT_FILE
                        path to the output file. If not specified, the output file will be the same as the restraints file with a .pkl extension.
```
To generate a `.pkl` file for GRASP inference, you need to provide two key files: a restraints file and a FASTA file.

### Restraints File Format
Each line of the restraints file contains one restraint, structured as follows:
#### RPR restraint:
Each line should contain two residues, the distance cutoff, and optionally, the false discovery rate (FDR). The format is: 

`residue1, residue2, distance_cutoff[, fdr]`

Here, `residue1` and `residue2` are formatted as `chain_index-residue_index-residue_type`, where:

`chain_index` is 1-indexed (referring to the position of the chain in the FASTA file),

`residue_index` is 1-indexed (the position of the residue within the chain),

`residue_type` is the single-letter amino acid code, which must match the corresponding entry in the FASTA file.

Example: "1-10-G" represents the 10th residue in the first chain, and the residue is Glycine (G).
The `distance_cutoff` specifies the maximum allowed distance between the two residues, and the `fdr` (optional, defaulting to 0.05) represents the false discovery rate for the RPR restraint.

#### IR Restraint:
For independent residue (IR) restraints, the format is simpler:

`residue`

The residue is also specified in the same format as above: `chain_index-residue_index-residue_type`.

### Example of a Restraints File:
```
1-10-G, 1-20-A, 8.0, 0.05
1-15-L
```
In this example, the first line represents an RPR restraint between the 10th Glycine (G) and the 20th Alanine (A) in chain 1, with a distance cutoff of 8.0Å and an FDR of 0.05. The second line is an IR restraint for the 15th Leucine (L) in chain 1.

The FASTA file contains the sequence of the protein complex. And it should be the one utilized for searching through the genetic database.

## Inference

### Arguments
To perform inference using GRASP, you must first prepare a feature dictionary utilizing AlphaFold-multimer's protocol. Alternatively, you can specify the required arguments to generate the feature dictionary. Below are some unique arguments specifically for GRASP inference:
```bash
run_grasp.py:
  --feature_pickle: Path to the feature dictionary generated using AlphaFold-
    multimer's protocal. If not specified, other arguments used for generating
    features will be required.
  --iter_num: Maximum iteration for iterative restraint filtering.
    (default: '5')
    (an integer)
  --mode: The mode of running GRASP, "normal" or "quick".
    (default: 'normal')
  --mrc_path: Path to the mrc file of density map.
  --output_dir: Path to a directory that will store the results.
  --resolution: Resolution of the mrc file
    (default: '10.0')
    (a number)
  --restraints_pickle: Path to a restraint pickle file. If not provided,
    inference will be done without restraints.
```
The other arguments are identical to those used in AlphaFold. You can view the complete list of arguments by typing `python run_grasp.py --help`.

### An example
```bash
unzip examples/1DE4.zip
python run_grasp.py \
   --feature_pickle examples/1DE4/features.pkl \
   --restraints_pickle examples/1DE4/IR_restr.pkl \
   --output_dir results
```


## Outputs

   The output directory will include following files:

   * ranked_*.pdb : pdb files ranked by plddt default.
   
   * unrelaxed_model_1_v3_v11_{ckpt}\_{seed}_{iter}.pdb: predicted structure in each iteration.

   * unrelaxed_model_1_v3_v11_{ckpt}\_{seed}_final.pdb: the predicted structure in the final iteration
     
   * unrelaxed_model_1_v3_v11_{ckpt}\_{seed}.csv: recorded crash and restraints satisfication of predicted structure in each iteration.

## Integrated modeling with Combift

   This part will be avaiable upon we prepared correspongding code.

## Dataset in GRASP paper

  You can download all datasets in GRASP original paper in the [link](https://osf.io/6kjuq/)

## Citations
If you use this package, please cite as the following:
```python
@article {Xie2024.09.16.613256,
	author = {Xie, Yuhao and Zhang, Chengwei and Li, Shimian and Du, Xinyu and Wang, Min and Hu, Yingtong and Liu, Sirui and Gao, Yi Qin},
	title = {Integrating various Experimental Information to Assist Protein Complex Structure Prediction by GRASP},
	elocation-id = {2024.09.16.613256},
	year = {2024},
	doi = {10.1101/2024.09.16.613256},
	publisher = {Cold Spring Harbor Laboratory},
	eprint = {https://www.biorxiv.org/content/early/2024/09/21/2024.09.16.613256.full.pdf},
	journal = {bioRxiv}


