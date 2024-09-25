"""
    Date: 07/10/2024
    Package Requirements:
    1.Powerfit3
    2.numpy==1.22.4
    3.Cython==0.29.33
    4.scikit-learn
    5.pdb-tools
    An implementation for Combfit heteroligomer algorithm with newest ranking function
"""
import os
import subprocess
import glob
import shutil
from powerfit import (
      Volume, Structure, structure_to_shape_like
      )
import numpy as np
import itertools
import time
from multiprocessing import Pool
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import sys
import argparse
import pdb
def ranking_function(LCC, CC, overlap):
    x = [LCC, CC, overlap]
    total_score = -1.10212834e+04 * x[0] + 3.45074839e-01 * x[1] + -6.68751777e-05 * x[2]
    # total_score = -1.10212834e+04 * x[0] + 3.45074839e-01 * x[1]
    return total_score

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

        # powerfit_prefix = '/lustre/grp/gyqlab/zhangcw/miniconda3/envs/colabfold_multimer/bin/powerfit'
        powerfit_prefix = 'powerfit'
        # pdb_split_chain_prefix = '/lustre/grp/gyqlab/zhangcw/miniconda3/envs/colabfold_multimer/bin/pdb_splitchain'
        pdb_split_chain_prefix = 'pdb_splitchain'
        # Split the pdb file into chains
        
        process = subprocess.Popen(f" {pdb_split_chain_prefix} {self.pdb_path}", cwd=self.output_path, shell=True)
        process.communicate()
        base_name = os.path.basename(self.pdb_path).split('.')[0]
        pdb_files = glob.glob(os.path.join(self.output_path, f'{base_name}_*.pdb'))
        # pdb_files = glob.glob(os.path.join(self.output_path, '*.pdb'))
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
        # import pdb;pdb.set_trace()
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

    def get_combine_result(self, pdbs_trans, overlap): # For each rigid body translation
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
        # std_mrc = np.std(self.normalized_mrc_array)
        # std_pdb = np.std(pdb_array)
        std_mrc = np.sqrt(((self.normalized_mrc_array - mean_mrc) ** 2).sum())
        std_pdb = np.sqrt(((pdb_array - mean_pdb) ** 2).sum())
        
        CC_score = ((self.normalized_mrc_array - mean_mrc) * (pdb_array - mean_pdb)).sum() / (std_mrc * std_pdb) # CC score
        non_zero_number = np.logical_and(self.normalized_mrc_array > 0, pdb_array > 0).sum()
        LCC_score = CC_score / non_zero_number
        total_score = ranking_function(LCC_score, CC_score, overlap)
        print(CC_score, LCC_score, non_zero_number, total_score)
        total_score = float(total_score)
        return total_score, whole_struct, pdbs_trans[1] # Return the whole structure and the score and combined label 
       
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
                score, whole_structure, label = self.get_combine_result((transformed_struct,combination), overlaps)
                if score > best_score:
                    print("converging ",score)
                    best_score = score
                    best_whole_struct = whole_structure
                    best_label = label
            try:
                self.best_label = best_label
            except:
                self.best_label = self.best_label
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
        
        # Genetic Algorithm
        new_search_label = []
        
        for i in range(self.n_chains):
            start_position = self.best_label[i] # 第i个链的最佳位置
            if start_position == 0:
                new_search_label.append([start_position])
            else:
                idx = max(0, start_position - self.n_chains * 2)
                new_search_label.append(list(range(0, idx)))
        
        best_score, structure = self.ranking_structure(new_search_label, best_score, structure)
        
        # Save the best structure
        best_structure_path = os.path.join(self.output_path, f'{self.pdb_name_used}_best_structure.pdb')
        structure.tofile(best_structure_path)
               

class Combfit_homomer(Combfit):
    def __init__(self, mrc_path, resolution, pdb_model_path,
                 output_path, gpu_flag, nproc_num, name, gt_path = None, skip=False, homomer_chains = 0, start_pos = 0, assigned_chain_num = -1):
        # Start position is where the representation of the homomer chain starts from
        super().__init__(mrc_path, resolution, pdb_model_path, output_path, gpu_flag, nproc_num, name, gt_path, skip)
        self.homomer_chains = homomer_chains
        self.start_pos = start_pos
        self.assigned_chain_num = assigned_chain_num
    
    def _rigid_body_docking_homomer(self):
        print(self.pdb_name_used)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        powerfit_prefix = '/lustre/grp/gyqlab/zhangcw/miniconda3/envs/colabfold_multimer/bin/powerfit'
        pdb_split_chain_prefix = '/lustre/grp/gyqlab/zhangcw/miniconda3/envs/colabfold_multimer/bin/pdb_splitchain'
        if not isinstance(self.pdb_path, tuple):
            if not self.skip:
                process = subprocess.Popen(f" {pdb_split_chain_prefix} {self.pdb_path}", cwd=self.output_path, shell=True)
                process.communicate()
        # base_name = os.path.basename(self.pdb_path).split('.')[0]
        # pdb_files = glob.glob(os.path.join(self.output_path, f'{base_name}_*.pdb'))
        pdb_files = glob.glob(os.path.join(self.output_path, '*.pdb'))
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
            temp_data['chain'] = np.full(temp_data['chain'].shape, chr(65 + _ + self.start_pos))
            new_structure = Structure(temp_data)
            self.homomer_structures.append(new_structure)
    
        
        print("Homomer Chain Numer:", self.homomer_chains)
        chains_string = ','.join([chr(65 + i + self.start_pos) for i in range(self.homomer_chains)])
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
        # print(cmd)
        if self.skip:
            return pdb_file
        else:
            self.run_cmd(cmd)
            return pdb_file
    
    
    def get_homomer_structures(self, pdb_file):
        monomer_structures = Structure.fromfile(pdb_file)
        structure_data = monomer_structures.data
        self.homomer_structures = []
        for _ in range(self.homomer_chains):
            from copy import deepcopy
            temp_data = deepcopy(structure_data)
            temp_data['chain'] = np.full(temp_data['chain'].shape, chr(65 + _ + self.start_pos))
            new_structure = Structure(temp_data)
            self.homomer_structures.append(new_structure)
    
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
            if self.assigned_chain_num != -1:
                kmeans = KMeans(n_clusters=self.assigned_chain_num, n_init=29, random_state=0).fit(data_to_cluster)
            else:
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
        
        # if self.assigned_chain_num != -1 and self.homomer_chains == 1:
        #     self.homomer_structures *= self.assigned_chain_num
            
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
                score, whole_structure, label = self.get_combine_result((transformed_struct,combination), overlaps)
                if score > best_score:
                    print("converging ",score)
                    best_score = score
                    best_whole_struct = whole_structure
                    best_label = label
            try:
                self.best_label = best_label
            except:
                self.best_label = self.best_label
        return best_score, best_whole_struct
    
    
    def get_ranking_combinations(self, combined_label):
        # new_search_space = []
        # for i in range(self.homomer_chains):
        #     # Extend to full search space
        #     new_search_space.append([self.homomer_raw_solution[i][j] for j in range(len(self.homomer_raw_solution[i]))])
        storage = set()
        for combination in itertools.product(*combined_label): # 聚类算法的结果
            # Judge whether it is unique
            if len(set(combination)) != len(combination):
                continue
            if tuple(set(combination)) in storage:
                continue
            else:
                storage.add(tuple(set(combination)))
        if len(storage) == 0:
            storage.add(tuple(set(itertools.product(*combined_label))))
        storage = list(storage)
        # storage = [i[0] for i in storage] # ?
        return list(storage)
    
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
            
        best_score, structure = self.ranking_structure_homomer(self.homomer_combined_label)
        # Genetic Algorithm
        new_search_label = []
        
        for i in range(self.homomer_chains):
            start_position = self.best_label[i] # 第i个链的最佳位置
            if start_position == 0:
                new_search_label.append([start_position])
            else:
                idx = max(0, start_position - self.homomer_chains * 2)
                new_search_label.append(list(range(idx, start_position)))
        
        best_score, structure = self.ranking_structure_homomer(new_search_label, best_score, structure)
        
        # Save the best structure
        best_structure_path = os.path.join(self.output_path, f'{self.pdb_name_used}_best_structure.pdb')
        structure.tofile(best_structure_path)

class Combfit_heteoligomer(Combfit):
    def __init__(self, mrc_path, resolution,
                 output_path, gpu_flag, nproc_num, name, gt_path = None, skip=False, homomer_chains = 0, start_pos = 0, pdb_paths = None, pdb_model_path = None):
        # Start position is where the representation of the homomer chain starts from
        super().__init__(mrc_path, resolution, pdb_model_path, output_path, gpu_flag, nproc_num, name, gt_path, skip)
        self.pdb = pdb_paths # 直接是已经拆分好的单链PDB文件，pdbs的格式[[entity1所有文件位置],[entity2的所有文件位置]...]，其中每一个链的位置都是tuple类型
        self.output_paths = [os.path.dirname(i[0]) for i in self.pdb]
        self.homomer_chainnum_list = [len(i) for i in self.pdb]
        self.tot_chain = sum(self.homomer_chainnum_list)
        temp = [sum(self.homomer_chainnum_list[:i]) for i in range(len(self.homomer_chainnum_list) + 1)]
        self.start_poses = temp[:-1]
    
    def ranking_structure_heteoligomer(self, labels, combfitter_list, best_score = -np.inf, best_whole_struct = None):
        """_summary_

        Args:
            labels (_type_): in the form of [[A...], [B...], [C...]]
                for example: A protien of A2B2 should have the form of [[(A11,A21), (A12,A22), (A13,A23)...], [(B11,B21), (B22,B22),...]]
            combfitter_list (_type_): the list of combfitter_homomer, where it stores structure and result
            best_score (_type_, optional): _description_. Defaults to -np.inf.
            best_whole_struct (_type_, optional): _description_. Defaults to None.
        """
        
        # To get a total combination we use itertools.product
        # pdb.set_trace()
        for heteoligomer_combination in itertools.product(*labels):
            # pdb.set_trace()
            transformed_struct = []
            # Get each combination
            for index, combination in enumerate(heteoligomer_combination):
                if isinstance(index, list):
                    index = index[0]
                combfitter = combfitter_list[index]
                struct_transform = [combfitter.homomer_result[i][num][0] for i, num in enumerate(combination)]
                temp_struct = [self.transform_structure(combfitter.homomer_structures[i], struct_transform[i]) for i in range(len(struct_transform))]
                transformed_struct += temp_struct
                
                
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
                score, whole_structure, label = self.get_combine_result((transformed_struct,combination), overlaps)
                if score > best_score:
                    print("converging ",score)
                    best_score = score
                    best_whole_struct = whole_structure
                    best_label = label
            try:
                self.best_label = best_label
            except:
                self.best_label = self.best_label
        return best_score, best_whole_struct
        
        
    
    def docking_pipeline_heteoligomer(self):
        combfitter_list = []
        for i in range(len(self.homomer_chainnum_list)):
            combfitter1 = Combfit_homomer(self.mrc_path, self.resolution, self.pdb[i], self.output_paths[i], self.use_gpu, self.nproc, self.pdb_name_used, self.ground_truth_path, self.skip, self.homomer_chainnum_list[i], self.start_poses[i], assigned_chain_num=self.tot_chain)
            combfitter_list.append(combfitter1)
        
        heteoligomer_files = []
        for index, i  in enumerate(combfitter_list):
            heteoligomer_files.append(i._rigid_body_docking_homomer())
            # heteoligomer_files.append(sorted(list(self.pdb[index]))[0])
            combfitter_list[index].get_homomer_structures(heteoligomer_files[index])
        
        self._preprocess()
        
        total_space = []
        labels = []
        for index in range(len(self.homomer_chainnum_list)):
            combfitter_list[index].KMeans_clustering_homomer(heteoligomer_files[index])
            total_space.append(combfitter_list[index].get_ranking_combinations(combfitter_list[index].homomer_combined_label))
            labels.append(combfitter_list[index].homomer_combined_label)
             
        best_score, structure = self.ranking_structure_heteoligomer(total_space, combfitter_list)
        best_structure_path = os.path.join(self.output_path, f'{self.pdb_name_used}_best_structure.pdb')
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
    parser.add_argument('-heteroligomer', action='store_true', help='Whether the target is a heteroligomer')
    # heteroligomer的文件路径和output在一个路径下 root path应该由文件夹构成，并且具有如下的结构root_1, root_2, root_3 ...
    args = parser.parse_args()
    
    if args.pdb:
        args.pdb = os.path.abspath(args.pdb)
    
    return args

def prepare_output_directory(path):
    if os.path.exists(path) and os.listdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def heteroligomer_list_construct(path):
    """_summary_
    To construct the pdb_paths of Combfit_heteoligomer
    Args:
        path (_type_): the root path to heteroligomer
    """
    
    pdb_paths = []
    for root, dirs, files in os.walk(path):
        folder1 = []
        for file in files:
            if file.endswith('.pdb'):
                folder1.append(os.path.join(root, file))
        if len(folder1) == 0:
            continue
        else:
            pdb_paths.append(tuple(folder1))
    
    return pdb_paths

if __name__=="__main__":
    args = get_args()
    
    if args.out and not args.skip and not args.heteroligomer:
        prepare_output_directory(args.out)
    
    if args.heteroligomer:
        pdb_paths = heteroligomer_list_construct(args.out) # output_path == heteroligomer_dir_path
        print(pdb_paths)
        combfitter = Combfit_heteoligomer(args.mrc, args.res, args.out, args.gpu, args.nproc, args.name, args.gt_path, args.skip, pdb_paths = pdb_paths)
        combfitter.docking_pipeline_heteoligomer()   
    elif args.homomer:
        combfitter = Combfit_homomer(args.mrc, args.res, args.pdb, args.out, args.gpu, args.nproc, args.name, args.gt_path, args.skip, args.homomer_num_chains)
        combfitter.docking_pipeline_homomer()
    else:
        combfitter = Combfit(args.mrc, args.res, args.pdb, args.out, args.gpu, args.nproc, args.name, args.gt_path, args.skip)
        combfitter.docking_pipeline()
