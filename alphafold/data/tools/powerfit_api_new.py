"""
    Package Requirements:
    1.Powerfit3
    2.numpy==1.22.4
    3.Cython==0.29.33
    4.scikit-learn
    5.pdb-tools
"""
import os
import subprocess
import sys
import glob
from powerfit import (
      Volume, Structure, structure_to_shape_like, proportional_orientations,
      quat_to_rotmat, determine_core_indices
      )
from powerfit.powerfitter import PowerFitter
from powerfit.analyzer import Analyzer
import numpy as np
import itertools
import time
from multiprocessing import Pool
from functools import partial
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import stat

def get_args():

    import argparse
    parser = argparse.ArgumentParser(description="powerfit pipeline")
    parser.add_argument("-mrc", help="mrc file location", default=None)
    parser.add_argument("-res", default=10.0, type=float, help="resolution")
    parser.add_argument("-pdb", default=None, type=str, help="pdb model file location")
    parser.add_argument("-out", default=None, type=str, help="output directory")
    parser.add_argument("-gpu", action="store_true", help="Whether to use GPU or not. Default is false")
    parser.add_argument("-nproc", default=1, type=int, help="Number of CPU cores to use. If GPU selected, the value needn't specify")
    args = parser.parse_args()
    return args

def get_combine_result(pdbs_trans, target, mrc_array, resolution, overlaps=0):
    start_time = time.time()
    pdbs = pdbs_trans[0]
    # Get the pdb arrays
    tot_data = pdbs[0].data
    for pdb in pdbs[1:]:
        tot_data = np.append(tot_data, pdb.data)
    whole_struct = Structure(tot_data)
    pdb_array = structure_to_shape_like(
            target, whole_struct.coor, resolution=resolution,
            weights= whole_struct.atomnumber, shape='vol'
            ).array

    # Penalize the struture exceeding the boundary
    print("stage1", time.time() - start_time)
    #mrc_array[mrc_array < 1e-2] = 0
    mean_mrc = np.average(mrc_array)
    mean_pdb = np.average(pdb_array)
    std_mrc = np.std(mrc_array)
    std_pdb = np.std(pdb_array)
    score = ((mrc_array - mean_mrc) * (pdb_array - mean_pdb)).sum() / (std_mrc * std_pdb)
    #score = (mrc_array[hit_grid] + pdb_array[hit_grid] - mrc_array[penal_1] / 10 - pdb_array[penal_2]).sum()
    print("combine result time", time.time() - start_time)
    return (score, pdbs, pdbs_trans[1], whole_struct)

def transform_structure(structure, sol):
    start_time = time.time()
    translated_structure = structure.duplicate()
    center = translated_structure.coor.mean(axis=1)
    translated_structure.translate(-center)
    out = translated_structure.duplicate()
    rot = np.asarray([float(x) for x in sol[7:]]).reshape(3, 3)
    trans = sol[4:7]
    out.rotate(rot)
    out.translate(trans)
    end_time = time.time()
    #print('transform structure time', end_time - start_time)
    return out

def calc_wether_overlap(struct1, struct2):
    start_time = time.time()
    coor1 = struct1.coor.T
    coor2 = struct2.coor.T
    tree1 = cKDTree(coor1)
    tree2 = cKDTree(coor2)
    overlap = tree1.query_ball_tree(tree2, r=1)
    overlap_count = sum(len(x) for x in overlap)
    end_time = time.time()
    # print("overlap check time", end_time - start_time)
    return True if (overlap_count / coor1.shape[0] > 0.05 or overlap_count /  coor2.shape[0] > 0.05) else overlap_count

def run_cmd(cmd):
    print(cmd)
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()
   
def normalize_map(map_data):
    """
        Normalize the mrc array based on the 95 percentile of nonzero elements.
        Inputs: a non-normalized mrc array
        Outputs: a normalized mrc array
    """
    try:
        percentile = np.percentile(map_data[np.nonzero(map_data)], 95)
        map_data /= percentile
    except IndexError as error:
        print("Cannot normalize the map")
     # set low valued data to 0
    print("### Setting all values < 0 to 0 ###")
    map_data[map_data < 0] = 0
    print("### Setting all values > 1 to 1 ###")
    map_data[map_data > 1] = 1
    
    return map_data
    
 
def docking_pipeline(mrc_path, resolution, pdb_path, output_path, name,use_gpu=False, nproc=1):
    """Runs the powerfit pipeline on the given input files.

    Args:
        mrcpath (str): Path to the mrc file.
        pdbpath (str): Path to the pdb file.
        output_path (str): Path to the output directory.
        temp_dir (str): Path to the temporary directory.
    """
    # Create the output directory if it does not exist
    start_time = time.time()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    powerfit_prefix = '/lustre/grp/gyqlab/zhangcw/miniconda3/envs/colabfold_multimer/bin/powerfit'
    pdb_split_chain_prefix = '/lustre/grp/gyqlab/zhangcw/miniconda3/envs/colabfold_multimer/bin/pdb_splitchain'
    # Split the pdb file into chains
    abs_pdb_path = os.path.abspath(pdb_path)
    process = subprocess.Popen(f" {pdb_split_chain_prefix} {abs_pdb_path}", cwd=output_path, shell=True)
    process.communicate()
    base_name = os.path.basename(pdb_path).split('.')[0]
    pdb_files = glob.glob(os.path.join(output_path, f'{base_name}_*.pdb'))
    pdb_files.sort() #? Sorted by Alphabetical order
    structures = [Structure.fromfile(pdb) for pdb in pdb_files]
    n_chains = len(pdb_files) 
    print("chains number", n_chains)
    out_pdbs = ''
    combined_label =[]
    result = []
    raw_solution = []
    cmd_tot = []
    for file in pdb_files:
        # Get pdb name
        pdb_name = os.path.basename(file).split('.')[0]
        
        # Create the output directory for the pdb
        powerfit_output_path = os.path.join(output_path, pdb_name)
        if not os.path.exists(powerfit_output_path):
            os.makedirs(powerfit_output_path)
        if not use_gpu:
            cmd = f"{powerfit_prefix} {mrc_path} {resolution} {file} -d {powerfit_output_path} -p {nproc} -a 10 -n 0"
        else:
            cmd = f"{powerfit_prefix} {mrc_path} {resolution} {file} -d {powerfit_output_path} -g -a 10 -n 0"
        print(cmd)
        cmd_tot.append(cmd)
    with Pool() as p:
        p.map(run_cmd, cmd_tot)
        p.close()
        p.join()

        
    for file in pdb_files:
        pdb_name = os.path.basename(file).split('.')[0]
        
        # Create the output directory for the pdb
        powerfit_output_path = os.path.join(output_path, pdb_name)
        solution_path = os.path.join(powerfit_output_path, 'solutions.out')
        with open(solution_path, 'r') as f:
            lines = f.readlines()
        lines = lines[1:]
        data_full = []
        for line in lines:
            data_list = [float(i) for i in line.split()]
            data_full.append(data_list)
            data_array = np.array(data_full)
        indices = int(len(data_array) * 0.4)
        if indices < n_chains:
            indices = n_chains
        data_array = data_array[:min(indices, 30)]
        raw_solution.append(data_array)
        
        data_to_cluster = data_array[:, 4:7]
        try:
            kmeans = KMeans(n_clusters=n_chains * 1, n_init=29, random_state=0).fit(data_to_cluster)
                
            labels = kmeans.labels_
        except:
            print("Kmeans failed")
            labels = []
            for i in range(len(data_to_cluster)):
                labels.append(i)
                print(labels)
        sorted_dict = {}
        fit_label = []
        for label in np.unique(labels):
            elements = data_array[labels == label]
            sorted_elements = elements[np.argsort(elements[:, 0])]
            sorted_dict[label] = sorted_elements
        fit_label = sorted(sorted_dict.keys())
        combined_label.append(fit_label)
        result.append(sorted_dict)
    
    # Select the best combination
    # Get the target volume
    target = Volume.fromfile(mrc_path)
    normalized_mrc_array = normalize_map(target.array)
    
    best_score = -np.inf
    best_pdb_names = None
    
    resonable_structures = []
    best_struct = None
    for combination in itertools.product(*combined_label):
        struct_transform = [result[i][num][0] for i,num in enumerate(combination)]
        transformed_struct = [transform_structure(structures[i], struct_transform[i]) for i in range(n_chains)]
        # Check Overlap
        overlap = False
        overlaps = 0
        for struct_comb in itertools.combinations(transformed_struct, 2):
            flag=calc_wether_overlap(struct_comb[0], struct_comb[1])
            if flag == True:
                overlap = True
                break
            else:
                overlaps += flag
        if overlap == True:
            continue
        else:
            # Mememorize non overlapping result
            resonable_structures.append((transformed_struct, struct_transform))
            score, struct, comb, whole_struct = get_combine_result((transformed_struct, struct_transform),target, normalized_mrc_array, resolution)
            if score > best_score:
                print("converging ",score)
                best_score = score
                best_struct = struct
                best_transform = comb
                best_whole_struct = whole_struct
    
    # Generate pdbs
    transformed_struct = best_struct
    best_whole_struct.tofile(os.path.join(output_path,f'{name}.pdb'))
    print(best_score)
    
    # Redocking search of adjacent space
    # Get the new_structure_space
    new_search_space = []
    struct_transform = best_transform
    for i in range(len(struct_transform)): # i is the chain index
        idx = int(struct_transform[i][0]) - 1
        # print(idx)
        if idx == 0:
            new_search_space.append([struct_transform[0]])
        else:
            start_position = max(0, idx - n_chains * 2)
            # full search space
            new_search_space.append([raw_solution[i][j] for j in range(0, start_position)])
    best_struct = None
    for transform_combination in itertools.product(*new_search_space):
        transformed_struct = [transform_structure(structures[i], transform_combination[i]) for i in range(n_chains)]
        # Check Overlap
        overlap = False
        overlaps = 0
        for struct_comb in itertools.combinations(transformed_struct, 2):
            flag=calc_wether_overlap(struct_comb[0], struct_comb[1])
            if flag == True:
                overlap = True
                break
            else:
                overlaps += flag
        if overlap == True:
            continue
        else:
            # No Overlap Calculate the Score
            score, _, _, whole_struct = get_combine_result((transformed_struct,0), target, normalized_mrc_array, resolution, overlaps)
            if score > best_score:
                print('converging ', score)
                best_score = score
                best_struct = transformed_struct
                best_whole_struct = whole_struct
   
   
    if best_whole_struct is not None: 
        print(best_score)
        # Generate pdbs
        best_whole_struct.tofile(os.path.join(output_path, f'{name}.pdb'))
    end_time = time.time()
    # Compare the best result with the original pdb
    new_output_path = os.path.join(powerfit_output_path, 'original')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    docking_cmd = f"{powerfit_prefix} {mrc_path} {resolution} {pdb_path} -d {new_output_path} -g -a 10 -n 1 -cw"
    run_cmd(docking_cmd)
    
    # Compare the best result with the original pdb
    fit_pdb = os.path.join(new_output_path, 'fit_1.pdb')
    original_struct = Structure.fromfile(fit_pdb)
    
    # Implement the get_combine_result
    pdb_array = structure_to_shape_like(
            target, original_struct.coor, resolution=resolution,
            weights= original_struct.atomnumber, shape='vol'
            ).array
    mean_mrc = np.average(normalized_mrc_array)
    mean_pdb = np.average(pdb_array)
    std_mrc = np.std(normalized_mrc_array)
    std_pdb = np.std(pdb_array)
    original_score = ((normalized_mrc_array - mean_mrc) * (pdb_array - mean_pdb)).sum() / (std_mrc * std_pdb)
    
    if original_score > best_score:
        best_score = original_score
        print('original win', original_score)
        best_whole_struct = original_struct
        best_whole_struct.tofile(os.path.join(output_path, f'{name}.pdb'))
    else:
        print('docking win')
    
    print(end_time - start_time)
    return best_score



if __name__=="__main__":
    args = get_args()

    docking_pipeline(args.mrc, args.res, args.pdb, args.out, args.gpu, args.nproc)

