"""
    Date: 05/15/2024
    Package Requirements:
    1.Powerfit3
    2.numpy==1.22.4
    3.Cython==0.29.33
    4.scikit-learn
    5.pdb-tools
    The program is an implementation of the powerfit pipeline for CC ranking algorithm
    The program is an implementation of Combfit algorithm used on homomers
    The program is used for data collection, which will create a folder containing all LCC,CC scores and corresponding intermediate pdb file
    The program does not contain the process of computing AF2Rank score, which shall be computed separately.
    To run the program, no gpu is needed. Please set the parameters to the number of CPU cores
"""
import os
import subprocess
import glob
import logging
from powerfit import (
      Volume, Structure,proportional_orientations,quat_to_rotmat, determine_core_indices
      )
from powerfit.powerfitter import PowerFitter
from powerfit.analyzer import Analyzer
from functools import partial
from scipy.spatial.distance import cdist
import numpy as np
import itertools
import time
from multiprocessing import Pool
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import sys
import argparse
class Combfit:
    def __init__(self, mrc_path, resolution, pdb_model_path,
                 output_path, gpu_flag, nproc_num, name, gt_path = None, skip=False):
        self.mrc_path = mrc_path
        self.resolution = resolution
        self.pdb_path = pdb_model_path
        self.output_path = output_path
        self.use_gpu = gpu_flag
        self.nproc = nproc_num
        self.pdb_name_used = name
        self.ground_truth_path = gt_path
        self.chains_string = None # chain string in alhpabetical order A,B,C,D...
        self.n_chains = 0
        self.structures = None # Powerfit structure format for each chain
        self.normalized_mrc_array = None
        self.skip = skip
        self.best_label = None
    
    def clear(self):
        self.best_label = None
        self.best_score = -np.inf
        self.best_structure = None
        self.skip = None
        self.structures = None
        self.result = None
    
    def run_cmd(self, cmd):
        print(cmd)
        process = subprocess.Popen(cmd, shell=True)
        process.communicate()
        
    def _rigid_body_docking(self):
        # Create the output directory if it does not exist
        print(self.pdb_name_used)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # python_prefix = os.environ['CONDA_PREFIX'] + '/bin/python'
        # powerfit_prefix = os.environ["CONDA_PREFIX"] + '/bin/powerfit'
        # pdb_split_chain_prefix = os.environ["CONDA_PREFIX"] + '/bin/pdb_splitchain'

        powerfit_prefix = '/lustre/grp/gyqlab/zhangcw/miniconda3/envs/colabfold_multimer/bin/powerfit'
        pdb_split_chain_prefix = '/lustre/grp/gyqlab/zhangcw/miniconda3/envs/colabfold_multimer/bin/pdb_splitchain'
        # Split the pdb file into chains
        pdb_path = os.path.abspath(self.pdb_path)
        process = subprocess.Popen(f" {pdb_split_chain_prefix} {pdb_path}", cwd=self.output_path, shell=True)
        process.communicate()
        base_name = os.path.basename(self.pdb_path).split('.')[0]
        pdb_files = glob.glob(os.path.join(self.output_path, f'{base_name}_*.pdb'))
        pdb_files.sort() #? Sorted by Alphabetical order
        structures = [Structure.fromfile(pdb) for pdb in pdb_files]
        self.structures = structures
        
        n_chains = len(pdb_files) 
        self.n_chains = n_chains
        print("chains number", n_chains)
        
        # construct the whole chain string from n_chains
        chains_string = ','.join([chr(65 + i) for i in range(n_chains)])
        self.chains_string = chains_string
        cmd_tot = []
        for file in pdb_files:
            # Get pdb name
            pdb_name = os.path.basename(file).split('.')[0]
            
            # Create the output directory for the pdb
            powerfit_output_path = os.path.join(self.output_path, pdb_name)
            if not os.path.exists(powerfit_output_path):
                os.makedirs(powerfit_output_path)
            if not self.use_gpu:
                cmd = f"{powerfit_prefix} {self.mrc_path} {self.resolution} {file} -d {powerfit_output_path} -p {self.nproc} -a 10 -n 0"
            else:
                cmd = f"{powerfit_prefix} {self.mrc_path} {self.resolution} {file} -d {powerfit_output_path} -g -a 10 -n 0"
            print(cmd)
            cmd_tot.append(cmd)
        if self.skip:
            return pdb_files
        else:
            with Pool() as p:
                p.map(self.run_cmd, cmd_tot)
                p.close()
                p.join()
        
            return pdb_files
    
    def normalize_map(self, map_data):
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
    
    def _preprocess(self):
        """
            A utility function to preprocess mrc file and target file
        """
        self.target = Volume.fromfile(self.mrc_path)
        normalized_mrc_array = self.normalize_map(self.target.array)
        self.normalized_mrc_array = normalized_mrc_array
        self.model = None
        self.best_score = -np.inf
        self.best_structure = None
    
    def KMeans_clustering(self, pdb_files):
        result = []
        combined_label = []
        raw_solution = [] 
        
        for file in pdb_files:
            pdb_name = os.path.basename(file).split('.')[0]
            
            # Create the output directory for the pdb
            powerfit_output_path = os.path.join(self.output_path, pdb_name)
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
            if indices < self.n_chains:
                indices = self.n_chains
            data_array = data_array[:min(indices, 30)]
            raw_solution.append(data_array)
            
            data_to_cluster = data_array[:, 4:7]
            try:
                kmeans = KMeans(n_clusters=self.n_chains * 1, n_init=29, random_state=0).fit(data_to_cluster)
                    
                labels = kmeans.labels_
            except:
                logging.info("Kmeans failed")
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
            
        self.result = result
        self.combined_label = combined_label
        self.raw_solution = raw_solution
    
    def transform_structure(self, structure, sol):
        translated_structure = structure.duplicate()
        center = translated_structure.coor.mean(axis=1)
        translated_structure.translate(-center)
        out = translated_structure.duplicate()
        rot = np.asarray([float(x) for x in sol[7:]]).reshape(3, 3)
        trans = sol[4:7]
        out.rotate(rot)
        out.translate(trans)
        #print('transform structure time', end_time - start_time)
        return out
    
    def calc_wether_overlap(self, struct1, struct2):
        coor1 = struct1.coor.T
        coor2 = struct2.coor.T
        tree1 = cKDTree(coor1)
        tree2 = cKDTree(coor2)
        overlap = tree1.query_ball_tree(tree2, r=1)
        overlap_count = sum(len(x) for x in overlap)
        min_distance = np.min(tree1.query(tree2.data, k = 1))
        
        flag = overlap_count / coor1.shape[0] > 0.3 or overlap_count /  coor2.shape[0] > 0.3
        return (flag, overlap_count, min_distance)
    
    def calc_TMscore(self, predict_path):
   
        native_path = self.ground_truth_path

        cmd = f"bash /lustre/grp/gyqlab/zhangcw/dxy/accurate_score_powerfit/20240512_data_collection/compute_TMscore.sh {native_path} {predict_path}"
        print(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        output, _ = process.communicate()
        print(output)
        try:
            output = float(output.decode().strip())
        except:
            output = 0
        return output

    def get_combine_result(self, pdbs_trans): # For each rigid body translation
        # Define a temporary directory 
        tempdir = os.path.join(self.output_path, 'collection')
        start_time = time.time()
        pdbs = pdbs_trans[0]
        # Get the pdb arrays
        tot_data = pdbs[0].data
        for pdb in pdbs[1:]:
            tot_data = np.append(tot_data, pdb.data)
        whole_struct = Structure(tot_data)
        pdb_array = structure_to_shape_like(
                self.target, whole_struct.coor, resolution=self.resolution,
                weights= whole_struct.atomnumber, shape='vol'
                ).array

        # Penalize the struture exceeding the boundary
        print("stage1", time.time() - start_time)
        #mrc_array[mrc_array < 1e-2] = 0
        mean_mrc = np.average(self.normalized_mrc_array)
        mean_pdb = np.average(pdb_array)
        std_mrc = np.std(self.normalized_mrc_array)
        std_pdb = np.std(pdb_array)
        CC_score = ((self.normalized_mrc_array - mean_mrc) * (pdb_array - mean_pdb)).sum() / (std_mrc * std_pdb) # CC score
        # non_zero_number = np.logical_and(self.normalized_mrc_array > 0, pdb_array > 0).sum()
        # LCC_score = CC_score / non_zero_number
        #score = (mrc_array[hit_grid] + pdb_array[hit_grid] - mrc_array[penal_1] / 10 - pdb_array[penal_2]).sum()
        # I need to obtain AF score here, should be included name and chains and structures
        # if not os.path.exists(tempdir):
        #     os.makedirs(tempdir)
        # NATIVE_PATH = os.path.join(tempdir, f'{LCC_score}_{CC_score}_temp.pdb')
        # whole_struct.tofile(NATIVE_PATH)
        # TMscore = self.calc_TMscore(predict_path = NATIVE_PATH)
        # document_file = os.path.join(tempdir, 'document.txt')
        # if not os.path.exists(document_file):
        #     with open(document_file, 'w') as f:
        #         pass
        # with open(document_file, 'a') as f:
        #     f.write(f'{LCC_score},{CC_score},{TMscore}\n')
        # print(self.pdb_name_used, NATIVE_PATH)
        return CC_score, whole_struct, pdbs_trans[1] # Return the whole structure and the score and combined label 
       
    def ranking_structure(self, combined_label, best_score = -np.inf, best_whole_struct = None):
        # resonable_structures = []
        for combination in itertools.product(*combined_label): # 聚类算法的结果
            struct_transform = [self.result[i][num][0] for i,num in enumerate(combination)]
            transformed_struct = [self.transform_structure(self.structures[i], struct_transform[i]) for i in range(self.n_chains)]
            # Check Overlap
            overlap = False
            overlaps = 0
            
            interact_dict = {}
            for struct_comb in itertools.combinations(transformed_struct, 2):
                flag = self.calc_wether_overlap(struct_comb[0], struct_comb[1])
                if flag[0] == True:
                    overlap = True
                    break
                else:
                    overlaps += flag[1]
                    interact_dict[struct_comb[0]] = min(interact_dict[struct_comb[0]], flag[2]) if struct_comb[0] in interact_dict else flag[2]
                    interact_dict[struct_comb[1]] = min(interact_dict[struct_comb[1]], flag[2]) if struct_comb[1] in interact_dict else flag[2]
            if overlap == True:
                continue
            else:
                interact_flag = True
                for monomer in interact_dict:
                    if interact_dict[monomer] > 8:
                        interact_flag = False
                        break
                
                if not interact_flag:
                    continue

                # Mememorize non overlapping result
                # resonable_structures.append((transformed_struct, struct_transform))
                score, whole_structure, label = self.get_combine_result((transformed_struct,combination))
                print("converging ",score,"label",label,"best_score",best_score)
                if score >= best_score:
                    # print("converging ",score,"label",label)
                    best_score = score
                    best_whole_struct = whole_structure
                    best_label = label
            self.best_label = best_label
        return best_score, best_whole_struct
        
    def docking_pipeline(self):
        """Runs the powerfit pipeline on the given input files.

        Args:
            mrcpath (str): Path to the mrc file.
            pdbpath (str): Path to the pdb file.
            output_path (str): Path to the output directory.
            temp_dir (str): Path to the temporary directory.
        """
        pdb_files = self._rigid_body_docking()
        # import os
        # os._exit(0)
        # # Final results storage 
        self._preprocess()
        
        # KMeans clustering
        self.KMeans_clustering(pdb_files)
        
        # Get the best structure
        new_search_space = []
        for i in range(self.n_chains):
            # Extend to full search space
            new_search_space.append([self.raw_solution[i][j] for j in range(len(self.raw_solution[i]))])
    
        best_score, structure = self.ranking_structure(self.combined_label)
        # print('complete first round')
        # Genetic Algorithm
        new_search_label = []
        
        for i in range(self.n_chains):
            start_position = self.best_label[i] # 第i个链的最佳位置
            if start_position == 0:
                new_search_label.append([start_position])
            else:
                idx = max(0, start_position - self.n_chains * 2)
                new_search_label.append(list(range(0, idx)))
        # print(new_search_label)
        best_score, structure = self.ranking_structure(new_search_label, best_score, structure)
        
        # Save the best structure
        best_structure_path = os.path.join(self.output_path, f'{self.pdb_name_used}_best.pdb')
        structure.tofile(best_structure_path)
               

class Combfit_homomer(Combfit):
    def __init__(self, mrc_path, resolution, pdb_model_path,
                 output_path, gpu_flag, nproc_num, name, gt_path = None, skip=False, homomer_chains = 0):
        super().__init__(mrc_path, resolution, pdb_model_path, output_path, gpu_flag, nproc_num, name, gt_path, skip)
        self.homomer_chains = homomer_chains
    
    def _rigid_body_docking_homomer(self):
        print(self.pdb_name_used)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        powerfit_prefix = '/lustre/grp/gyqlab/zhangcw/miniconda3/envs/colabfold_multimer/bin/powerfit'
        pdb_split_chain_prefix = '/lustre/grp/gyqlab/zhangcw/miniconda3/envs/colabfold_multimer/bin/pdb_splitchain'
        pdb_path = os.path.abspath(self.pdb_path)
        if not self.skip:
            process = subprocess.Popen(f" {pdb_split_chain_prefix} {pdb_path}", cwd=self.output_path, shell=True)
            process.communicate()
        base_name = os.path.basename(self.pdb_path).split('.')[0]
        pdb_files = glob.glob(os.path.join(self.output_path, f'{base_name}_*.pdb'))
        pdb_files.sort()
        # For homomers splited pdbs are only needed to be considered once
        if not self.skip:
            self.homomer_chains = len(pdb_files)
        pdb_file = pdb_files[0] # Get the first chain to dock
        monomer_structures = Structure.fromfile(pdb_file)
        structure_data = monomer_structures.data
        self.homomer_structures = []
        for _ in range(self.homomer_chains):
            from copy import deepcopy
            temp_data = deepcopy(structure_data)
            temp_data['chain'] = np.full(temp_data['chain'].shape, chr(65 + _))
            new_structure = Structure(temp_data)
            self.homomer_structures.append(new_structure)
    
        
        print("Homomer Chain Numer:", self.homomer_chains)
        chains_string = ','.join([chr(65 + i) for i in range(self.homomer_chains)])
        self.homomer_chains_string = chains_string
        
        # Rigid docking with powerfit
        pdb_name = os.path.basename(pdb_file).split('.')[0]
        powerfit_output_path = os.path.join(self.output_path, pdb_name)
        if not os.path.exists(powerfit_output_path):
                os.makedirs(powerfit_output_path)
                
        # Decide grid angle
        angle = 10 / (self.homomer_chains ** (1/3))
        if not self.use_gpu:
            cmd = f"{powerfit_prefix} {self.mrc_path} {self.resolution} {pdb_file} -d {powerfit_output_path} -p {self.nproc} -a {angle} -n 0"
        else:
            cmd = f"{powerfit_prefix} {self.mrc_path} {self.resolution} {pdb_file} -d {powerfit_output_path} -g -a {angle} -n 0"
        print(cmd)
        if self.skip:
            return pdb_file
        else:
            self.run_cmd(cmd)
            return pdb_file
    
    def KMeans_clustering_homomer(self, file):
        result = []
        combined_label = []
        raw_solution = []
        pdb_name = os.path.basename(file).split('.')[0]
        powerfit_output_path = os.path.join(self.output_path, pdb_name)
        solution_path = os.path.join(powerfit_output_path, 'solutions.out')
        with open(solution_path, 'r') as f:
            lines = f.readlines()
        lines = lines[1:]
        data_full = []
        for line in lines:
            data_list = [float(i) for i in line.split()]
            data_full.append(data_list)
            data_array = np.array(data_full)
            
        proportion = min(1, 0.4 * self.homomer_chains)   
        indices = int(len(data_array) * proportion)
        if indices < self.homomer_chains:
            indices = self.homomer_chains
        data_array = data_array[:min(indices, 30 * self.homomer_chains)]
        raw_solution = [data_array for _ in range(self.homomer_chains)]
        
        data_to_cluster = data_array[:, 4:7]
        try:
            kmeans = KMeans(n_clusters=int(self.homomer_chains * 1.5), n_init=29, random_state=0).fit(data_to_cluster)
                
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
        combined_label = [fit_label for _ in range(self.homomer_chains)]
        result = [sorted_dict for _ in range(self.homomer_chains)]
            
        self.homomer_result = result
        self.homomer_combined_label = combined_label
        self.homomer_raw_solution = raw_solution
    
    def ranking_structure_homomer(self, combined_label, best_score = -np.inf, best_whole_struct = None):
        # resonable_structures = []
        storage = set()
        for combination in itertools.product(*combined_label): # 聚类算法的结果
            # Judge whether it is unique
            if len(set(combination)) != len(combination):
                continue
            if tuple(set(combination)) in storage:
                continue
            else:
                storage.add(tuple(set(combination)))
            struct_transform = [self.homomer_result[i][num][0] for i,num in enumerate(combination)]
            transformed_struct = [self.transform_structure(self.homomer_structures[i], struct_transform[i]) for i in range(self.homomer_chains)]
            # Check Overlap
            overlap = False
            overlaps = 0
            interact_dict ={}
            for struct_comb in itertools.combinations(transformed_struct, 2):
                flag = self.calc_wether_overlap(struct_comb[0], struct_comb[1])
                if flag[0] == True:
                    overlap = True
                    break
                else:
                    overlaps += flag[1]
                    interact_dict[struct_comb[0]] = min(interact_dict[struct_comb[0]], flag[2]) if struct_comb[0] in interact_dict else flag[2]
                    interact_dict[struct_comb[1]] = min(interact_dict[struct_comb[1]], flag[2]) if struct_comb[1] in interact_dict else flag[2]
            if overlap == True:
                continue
            else:
                interact_flag = True
                for monomer in interact_dict:
                    if interact_dict[monomer] > 8:
                        interact_flag = False
                        break
                
                if not interact_flag:
                    continue

                # Mememorize non overlapping result
                # resonable_structures.append((transformed_struct, struct_transform))
                score, whole_structure, label = self.get_combine_result((transformed_struct,combination))
                if score > best_score:
                    print("converging ",score)
                    best_score = score
                    best_whole_struct = whole_structure
                    best_label = label
            self.best_label = best_label
        return best_score, best_whole_struct
    
    
    def docking_pipeline_homomer(self):
        """Run the combfit pipeline on homomers input files
        Args:
            mrcpath (str): Path to the mrc file.
            pdbpath (str): Path to the pdb file.
            output_path (str): Path to the output directory.
            temp_dir (str): Path to the temporary directory.
            homomer_chains (int): Number of homomer chains
        """
        pdb_file = self._rigid_body_docking_homomer()
        self._preprocess()
        
        self.KMeans_clustering_homomer(pdb_file)
        
        new_search_space = []
        for i in range(self.homomer_chains):
            # Extend to full search space
            new_search_space.append([self.homomer_raw_solution[i][j] for j in range(len(self.homomer_raw_solution[i]))])
            
        best_score, structure = self.ranking_structure_homomer(self.homomer_combined_label)
        # Genetic Algorithm
        # new_search_label = []
        
        # for i in range(self.n_chains):
        #     start_position = self.best_label[i] # 第i个链的最佳位置
        #     if start_position == 0:
        #         new_search_label.append([start_position])
        #     else:
        #         idx = max(0, start_position - self.homomer_chains * 2)
        #         new_search_label.append(list(range(0, idx)))
        
        # best_score, structure = self.ranking_structure_homomer(new_search_label, best_score, structure)
        
        # Save the best structure
        best_structure_path = os.path.join(self.output_path, f'{self.pdb_name_used}_best.pdb')
        structure.tofile(best_structure_path)
        
        
def get_args():
    parser = argparse.ArgumentParser(description="powerfit pipeline")
    parser.add_argument("-mrc", help="mrc file location", default=None)
    parser.add_argument("-res", default=10.0, type=float, help="resolution")
    parser.add_argument("-pdb", default=None, type=str, help="pdb model file location")
    parser.add_argument("-out", default=None, type=str, help="output directory")
    parser.add_argument("-gpu", action="store_true", help="Whether to use GPU or not. Default is false")
    parser.add_argument("-nproc", default=1, type=int, help="Number of CPU cores to use. If GPU selected, the value needn't specify")
    parser.add_argument('-name', default=None, type=str, help="Name of the protein")
    parser.add_argument('-gt_path', default=None, type=str, help='If any, the path to ground truth')
    parser.add_argument('-skip', action='store_true', help='Whether to skip the process of powerfit')
    parser.add_argument('-homomer', action='store_true', help='Whether to use homomer mode of Combfit')
    parser.add_argument('-homomer_num_chains', default=0, type=int, help='Number of homomer chains, needed to be specify when skipped')
    args = parser.parse_args()
    return args

# if __name__=="__main__":
#     args = get_args()
#     if args.homomer:
#         combfitter = Combfit_homomer(args.mrc, args.res, args.pdb, args.out, args.gpu, args.nproc, args.name, args.gt_path, args.skip, args.homomer_num_chains)
#         combfitter.docking_pipeline_homomer()
#     else:
#         combfitter = Combfit(args.mrc, args.res, args.pdb, args.out, args.gpu, args.nproc, args.name, args.gt_path, args.skip)
#         combfitter.docking_pipeline()
