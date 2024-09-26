# GRASP-JAX

A tool for accurate protein complex structure prediction integrated with diverse experimental restraints. (JAX version)

## Installation

If you have installed the AlphaFold in your machine and have downloaded the dataset, you can just follow these steps:

1. Clone this respository , `cd` into it.

   ```bash
   git clone https://github.com/xiergo/GRASP_JAX.git
   cd ./GRASP_JAX
   ```

1. Download GRASP model weights from (https://osf.io/6kjuq/) and move them to the path where you stored the AlphaFold dataset

    ```
    cp params_model_1_v3_v11_*.npz PATH_OF_ALPHAFOLD_DATASET/params/
    ```
If you have not installed AlphaFold before, you can create an virtual Enviroment as this:

   ```
   conda create -n GRASP python=3.8 -f requiremts.txt
   ```

Then, you need download the required database as in the [AlphaFold](https://github.com/google-deepmind/alphafold) and move the model weights as above

## Inference

We provide a script to transform experimental restraints to the input pkl file for GRASP

   ```
   施工中
   ```

Then you can run GRASP inference as following command(e.g. 1DE4 in examples folder ):

   ```
   unzip examples/1DE4.zip  
   bash run_grasp.sh -d PATH_to_your_AF2_database -o test_1DE4 -f examples/1DE4.fasta \
   -t 2024-05-01 -q example/1DE4/RPR_restr.pkl
   ```
Explanation of arguments:

   * -d:  the path of AlphaFold databse
   * -o:  the output folder of inference
   * -f:  the path of fasta file for protein complex
   * -t:  cut-off date for avaliable monomer template
   * -q:  input restraints file

Extra optinal arguments:

   * -p:  Reuse the multiple sequence alignment(MSA), you need to prepare a folder named "msas" under the output folder
   * -z:  If you have the feature pickle file from AlphaFold, you can skip the process of querying msas and templates, must be sure the `-p` is true
   * -i:  Adjust the iteration num of iteractive filtering strategy by this argument, default is 1
   * -w:  Select inference mode of GRASP, 'quick' or 'normal', default mode is 'normal' 

Any other detailed argument can see the file run_alphafold.sh and run_alphafold.py. A common adjusted argument is `max_recycle_per_iter` which
controls the max recycle number in each iteration. You can seach this argument in the run_alphafold.py. Default recycle number for normal mode 
is 20 and quick mode is 4.

## Output Directory

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


