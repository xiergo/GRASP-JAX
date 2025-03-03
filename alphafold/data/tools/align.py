# Ref: https://github.com/sokrypton/ColabDesign.git
import os
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Superimposer,PPBuilder
from alphafold.common import protein
from scipy.spatial.distance import cdist
import random
from Bio import PDB
from sklearn.cluster import KMeans
from Bio.PDB.Polypeptide import three_to_one
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'


def interpolate_structures(structure_A, structure_B, t ,output_path):
   parser = PDBParser()
   structure_A = parser.get_structure('A', structure_A)
   structure_B = parser.get_structure('B', structure_B)
   aligned_A, aligned_B = align_structures(structure_A, structure_B)
   interpolated_structure = interpolate(aligned_A, aligned_B, t)
   io = PDBIO()
   io.set_structure(interpolated_structure)
   io.save(output_path)
   
def align_structures(structure_A, structure_B):
    sup = Superimposer()
    atoms_A = [atom for atom in structure_A.get_atoms() if atom.get_name() == 'CA']
    atoms_B = [atom for atom in structure_B.get_atoms() if atom.get_name() == 'CA']
    
    sup.set_atoms(atoms_A, atoms_B)
    sup.apply(structure_B.get_atoms())
    
    return structure_A, structure_B

def interpolate(structure_A, structure_B, t):
    coords_A = np.array([atom.get_coord() for atom in structure_A.get_atoms()])
    coords_B = np.array([atom.get_coord() for atom in structure_B.get_atoms()])
    
    interpolated_coords = (1 - t) * coords_A + t * coords_B

    for i, atom in enumerate(structure_A.get_atoms()):
        atom.set_coord(interpolated_coords[i])
    
    return structure_A
  

def max_min_sampling(data, k):
    selected_indices = [np.random.randint(data.shape[0])]
    selected_points = [data[selected_indices[-1]]]
    for _ in range(1, k):
        distances = [np.min([np.linalg.norm(data[i] - point) for point in selected_points]) for i in range(data.shape[0])]
        next_index = np.argmax(distances)
        selected_indices.append(next_index)
        selected_points.append(data[next_index])
    return data[selected_indices]
  
def get_distri(dis,fdr=0.05):
    BINS = np.append(np.arange(4,33,1),np.inf)
    bin_ = np.ceil(dis) - 3
    x = np.ones((len(BINS)))
    x_bool = (BINS <= bin_+3) * (BINS >= bin_-3)
    x[x_bool] = (1-fdr)/x_bool.sum()
    x[~x_bool] = fdr/(len(BINS)-x_bool.sum())
    return x
  
def get_far_distri(dis,fdr=0.05):
    BINS = np.append(np.arange(4,33,1),np.inf)
    # bin_ = np.ceil(dis) - 3
    x = np.ones((len(BINS)))
    x_bool = (BINS > 32)
    x[x_bool] = (1-fdr)/x_bool.sum()
    x[~x_bool] = fdr/(len(BINS)-x_bool.sum())
    return x
  
def monte_carlo_deduplication(elements, min_diff=3, attempts=1000):
    best_result = []
    unique_elements = list(set(elements))
    
    for _ in range(attempts):
        np.random.shuffle(unique_elements)
        current_result = []
        
        for element in unique_elements:
            if not current_result or all(abs(element - x) > min_diff for x in current_result):
                current_result.append(element)
        
        if len(current_result) > len(best_result):
            best_result = current_result
    
    return best_result
    
# def extract_restraints(ori_pdb,new_pdb,max_1v1,restraints):
#     pdb_pre = protein.from_pdb_string(open(ori_pdb).read())
#     pdb_post = protein.from_pdb_string(open(new_pdb).read())
#     distance_pre = cdist(pdb_pre.atom_positions[:,1], pdb_pre.atom_positions[:,1]) + np.eye(pdb_pre.aatype.shape[0]) * 1e3
#     distance_pre_beta = cdist(pdb_post.atom_positions[:,3], pdb_post.atom_positions[:,3]) + np.eye(pdb_post.aatype.shape[0]) * 1e3
#     distance_post = cdist(pdb_post.atom_positions[:,1], pdb_post.atom_positions[:,1]) + np.eye(pdb_post.aatype.shape[0]) * 1e3
#     distance_post_beta = cdist(pdb_post.atom_positions[:,3], pdb_post.atom_positions[:,3]) + np.eye(pdb_post.aatype.shape[0]) * 1e3
#     mask = np.zeros_like(distance_pre, dtype=bool)
#     mask[(pdb_pre.chain_index[None,:] != pdb_pre.chain_index[:,None])] = 1
#     valid =  mask * (distance_pre >= 25) * (distance_post <= 15)
#     res_pool = np.where(valid != 0)
#     if len(res_pool[0]) != np.array([]):
#       sel_res = max_min_sampling(np.array(res_pool).T, max_1v1)
#       for pair in sel_res:
#           # print(pair)
#           restraints['sbr'][pair[0],pair[1]] = get_distri(distance_post[pair[0],pair[1]])
#           restraints['sbr'][pair[1],pair[0]] = get_distri(distance_post[pair[0],pair[1]])
#           restraints['sbr_mask'][pair[1],pair[0]] = 1
#           restraints['sbr_mask'][pair[0],pair[1]] = 1
#     row,column = np.where((distance_post_beta <= 8) * mask * (distance_post_beta >= 4))
#     interface = np.array([i for i in set(row.tolist() + column.tolist())])
#     deduplicated_interface = monte_carlo_deduplication(interface, min_diff=3)
#     restraints['interface_mask'][deduplicated_interface] = 1
#     return restraints
def extract_restraints(ori_pdb,new_pdb,max_1v1,feat):
  pdb_pre = protein.from_pdb_string(open(ori_pdb).read())
  pdb_post = protein.from_pdb_string(open(new_pdb).read())
  distance_pre = cdist(pdb_pre.atom_positions[:,1], pdb_pre.atom_positions[:,1]) + np.eye(pdb_pre.aatype.shape[0]) * 1e3
  # distance_pre_beta = cdist(pdb_post.atom_positions[:,3], pdb_post.atom_positions[:,3]) + np.eye(pdb_post.aatype.shape[0]) * 1e3
  distance_post = cdist(pdb_post.atom_positions[:,1], pdb_post.atom_positions[:,1]) + np.eye(pdb_post.aatype.shape[0]) * 1e3
#   distance_post_beta = cdist(pdb_post.atom_positions[:,3], pdb_post.atom_positions[:,3]) + np.eye(pdb_post.aatype.shape[0]) * 1e3
  chain_mask = np.zeros_like(distance_pre, dtype=bool)
  chain_mask[(pdb_pre.chain_index[None,:] != pdb_pre.chain_index[:,None])] = 1
  valid_close = chain_mask * (distance_pre - distance_post ) * (distance_post <= 25) * (distance_pre - distance_post >= 10)
#   res_pool_close = np.argsort(valid_close.flatten())[::-1][:min(500,np.sum(valid_close > 0))]
  res_pool_close = np.argsort(valid_close.flatten())[::-1][:np.sum(valid_close > 0)]
  res_pool_close = np.array([np.unravel_index(i, valid_close.shape) for i in res_pool_close])
#   sample_res = {}
  if len(res_pool_close[0]) != 0:
      sel_res = max_min_sampling(res_pool_close, min(max_1v1,len(res_pool_close)))
      for pair in sel_res:
          feat['sbr'][pair[0],pair[1]] = get_distri(distance_post[pair[0],pair[1]])
          feat['sbr'][pair[1],pair[0]] = get_distri(distance_post[pair[0],pair[1]])
          feat['sbr_mask'][pair[1],pair[0]] = 1
          feat['sbr_mask'][pair[0],pair[1]] = 1
        #   sample_res[pair] = distance_post[pair[0],pair[1]]
#   row,column = np.where((distance_post_beta <= 8) * chain_mask )
#   interface = np.array([i for i in set(row.tolist() + column.tolist())])
#   deduplicated_interface = monte_carlo_deduplication(interface, min_diff=3)
#   feat['interface_mask'][deduplicated_interface] = 1
#   valid_far  = chain_mask * (distance_post - distance_pre ) * (distance_post > 30)
#   res_pool_far = np.argsort(valid_far.flatten())[::-1][:min(200,np.sum(valid_far > 0))]
#   res_pool_far = np.array([np.unravel_index(i, valid_far.shape) for i in res_pool_far])
#   if len(res_pool_far[0]) != 0:
#       sel_res = max_min_sampling(res_pool_far, min(max_1v1,len(res_pool_far)))
#       for pair in sel_res:
#           feat['sbr'][pair[0],pair[1]] = get_far_distri(distance_post[pair[0],pair[1]])
#           feat['sbr'][pair[1],pair[0]] = get_far_distri(distance_post[pair[0],pair[1]])
#           feat['sbr_mask'][pair[1],pair[0]] = 1
#           feat['sbr_mask'][pair[0],pair[1]] = 1
  return feat


def compare_chains(chain1, chain2):
    """Compare two chains to see if they are identical."""
    if len(chain1) != len(chain2):
        return False
    flag = True
    for res1, res2 in zip(chain1, chain2):
        if res1.get_resname() != res2.get_resname() or len(res1) != len(res2):
            flag = False
    if flag == True:
        return True
    else:
        # Compare chains in reverse direction
        for res1, res2 in zip(chain1[::-1], chain2):
            if res1.get_resname() != res2.get_resname() or len(res1) != len(res2):
                return False
        return True

def group_chains(structure):
    """Group identical chains together."""
    chains = list(structure.get_chains())
    grouped = []
    while chains:
        chain = chains.pop(0)
        group = [chain]
        for other_chain in chains[:]:
            if compare_chains(chain, other_chain):
                group.append(other_chain)
                chains.remove(other_chain)
        grouped.append(group)
    return grouped

def reorder_chains(name, grouped_chains):
    """Reorder chains to make identical chains adjacent."""
    chain_id = 'A'
    count_num = 1
    for group in grouped_chains:
        folder_name = f'{name}_{count_num}'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for chain in group:
            chain.id = chain_id
            output_file = f"{folder_name}/{name}_{chain_id}.pdb"
            io = PDBIO()
            io.set_structure(chain)
            io.save(output_file)
            chain_id = chr(ord(chain_id) + 1)
        count_num += 1
            
def splitchain(input_file, name, output_dir):
    parser = PDBParser()
    # import pdb;pdb.set_trace()
    structure = parser.get_structure(name, input_file)
    grouped_chains = group_chains(structure)
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    os.chdir(output_dir)
    reorder_chains(name, grouped_chains)

def get_chain_sequences(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('structure', pdb_file)
    chain_sequences = {}
    for model in structure:
        for chain in model:
            sequence = []
            for residue in chain:
                resname = residue.get_resname().title()
                aa = three_to_one(resname.upper())
                sequence.append(aa)
            chain_sequences[chain.id] = ''.join(sequence)
    return chain_sequences

def reorder_and_renumber_chains(input_pdb, output_pdb, reference_sequences):
    parser = PDBParser()
    structure = parser.get_structure('input', input_pdb)
    model = structure[0]

    input_sequences = get_chain_sequences(input_pdb)
    new_structure = model.copy()
    new_structure.detach_parent()

    for chain in list(new_structure):
        new_structure.detach_child(chain.id)

    sequence_to_chain = {seq: chain_id for chain_id, seq in input_sequences.items()}
    
    new_chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    new_chain_index = 0
    
    for ref_seq in reference_sequences:
        chain_id = sequence_to_chain.get(ref_seq)
        if chain_id:
            new_chain = model[chain_id].copy()
            new_chain.id = new_chain_ids[new_chain_index]
            new_structure.add(new_chain)
            new_chain_index += 1

    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)

def renumber_chains(input_pdb, output_pdb):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', input_pdb)
    
    # Get the model (assuming only one model)
    model = structure[0]
    
    # Create a list of new chain IDs starting from 'A'
    new_chain_ids = [PDB_CHAIN_IDS[i] for i in range(len(list(model.get_chains())))]  # 65 is ASCII code for 'A'
    
    # Map old chain IDs to new chain IDs
    old_to_new_chain_ids = {}
    for old_chain_id, new_chain_id in zip(model.get_chains(), new_chain_ids):
        old_to_new_chain_ids[old_chain_id.id] = new_chain_id
    
    # Create a new structure with renumbered chains
    new_structure = model.copy()
    new_structure.detach_parent()
    
    # Clear existing chains
    for chain in list(new_structure):
        new_structure.detach_child(chain.id)
    
    # Add chains with new IDs
    for old_chain_id, new_chain_id in old_to_new_chain_ids.items():
        chain = model[old_chain_id]
        chain.id = new_chain_id
        new_structure.add(chain)
    
    # Save the renumbered structure to a new PDB file
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)
    
def restore_chains_from_merge(merged_pdb_file, residue_mapping, output_file):
    """
    根据 residue_mapping 将合并后的 PDB 文件恢复成原来的链结构
    """
    
    # 解析合并后的 PDB 文件
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('merged_protein', merged_pdb_file)
    
    # 创建新的结构用于保存还原后的链
    restored_structure = PDB.Structure.Structure("restored_structure")
    new_model = PDB.Model.Model(0)  # 所有链保存在同一个模型中
    
    # 创建一个空字典，用于存储还原后的链
    restored_chains = {}
    
    # 遍历合并后的链，并根据 residue_mapping 还原
    for new_chain in structure.get_chains():
        chain_id = new_chain.id
        if chain_id in residue_mapping:
            # 遍历合并链中的残基，根据 mapping 将其恢复到原始链和原始编号
            for entry in residue_mapping[chain_id]:
                original_chain_id = entry["original_chain"]
                original_residue_id = entry["original_residue"]
                new_residue_number = entry["new_residue_number"]
                
                # 提取合并链中的残基
                new_residue = next(r for r in new_chain.get_residues() if r.id == (' ', new_residue_number, ' '))
                
                # 如果原始链不存在，则创建
                if original_chain_id not in restored_chains:
                    restored_chains[original_chain_id] = PDB.Chain.Chain(original_chain_id)
                
                # 将残基恢复到原始编号，并添加到对应的原始链中
                restored_residue = new_residue.copy()
                restored_residue.id = original_residue_id  # 恢复原始残基编号
                restored_chains[original_chain_id].add(restored_residue)
    
    # 将所有恢复的链添加到模型中
    for chain in restored_chains.values():
        new_model.add(chain)
    
    # 添加模型到结构
    restored_structure.add(new_model)
    
    # 使用 PDBIO 将还原后的结构保存为新的 PDB 文件
    io = PDB.PDBIO()
    io.set_structure(restored_structure)
    io.save(output_file)


def parse_pdb_chains(pdb_file):
    """解析 PDB 文件并返回每个链的坐标和ID"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    chains = []
    
    for model in structure:
        for chain in model:
            atoms = list(chain.get_atoms())
            if len(atoms) > 0:
                # 计算链的几何中心 (质心)
                coords = np.array([atom.get_coord() for atom in atoms])
                centroid = np.mean(coords, axis=0)
                chains.append((chain.id, centroid))
    
    return chains

def random_group_by_proximity(chains, n_groups):
    """根据空间临近性将链分组"""
    # 提取所有链的质心坐标
    centroids = np.array([chain[1] for chain in chains])
    
    # 使用 KMeans 聚类，将链基于空间距离进行分组
    kmeans = KMeans(n_clusters=n_groups, random_state=42).fit(centroids)
    labels = kmeans.labels_
    
    # 随机打乱分组
    group_dict = {i: [] for i in range(n_groups)}
    for idx, (chain_id, _) in enumerate(chains):
        group_dict[labels[idx]].append(chain_id)
    
    # 随机化分组
    group_ids = list(group_dict.keys())
    random.shuffle(group_ids)
    
    shuffled_groups = {group_id: group_dict[group_id] for group_id in group_ids}
    
    return shuffled_groups

def merge_chains_by_group(pdb_file, grouped_chains, output_file="merged_output.pdb"):
    """
    将分组内的链合并为一个链，存储在同一个模型内，并返回残基编号映射，
    方便将合并后的链还原回原始结构。
    """
    
    # 解析 PDB 文件
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # 创建新的结构用于保存合并后的链
    merged_structure = PDB.Structure.Structure("merged_structure")
    
    # 用于唯一模型编号
    new_model = PDB.Model.Model(0)  # 所有链保存在同一个模型中
    
    # 存储原始链的残基编号映射，方便还原
    residue_mapping = {}
    
    # 遍历每个组，将组内的链合并到一个新的链
    for group_id, chains in grouped_chains.items():
        # 给每个组一个唯一的链名（A, B, C...）
        new_chain_name = chr(65 + group_id)  # A, B, C, ...
        merged_chain = PDB.Chain.Chain(new_chain_name)
        residue_mapping[new_chain_name] = []  # 存储合并链的残基编号映射
        
        # 用于重新编号残基序号
        new_residue_number = 1
        
        for chain_id in chains:
            for model in structure:
                for chain in model:
                    if chain.id == chain_id:
                        # 遍历该链的所有残基并添加到新的合并链中
                        for residue in chain.get_residues():
                            # 复制残基并更新残基序号，确保编号连续且唯一
                            new_residue = residue.copy()
                            new_residue.id = (' ', new_residue_number, ' ')
                            merged_chain.add(new_residue)
                            
                            # 保存原始链的残基编号和新的编号映射
                            residue_mapping[new_chain_name].append({
                                "original_chain": chain_id,
                                "original_residue": residue.id,
                                "new_residue_number": new_residue_number
                            })
                            
                            # 增加新的残基序号
                            new_residue_number += 1
        
        # 将合并后的链添加到同一个模型中
        new_model.add(merged_chain)
    
    # 添加模型到结构
    merged_structure.add(new_model)
    
    # 使用 PDBIO 将合并后的链保存为新的 PDB 文件
    io = PDB.PDBIO()
    io.set_structure(merged_structure)
    io.save(output_file)
    
    # 返回残基编号映射信息，方便后续还原
    return residue_mapping  

def chain_mapping(ref_str,target_str):
    parser = PDB.PDBParser(QUIET=True)
    reference_structure = parser.get_structure('reference', ref_str)
    target_structure = parser.get_structure('target',target_str)
    reference_chain_ids = [chain.id for chain in reference_structure.get_chains()]
    target_chain_ids = [chain.id for chain in target_structure.get_chains()]
    return {id:target_chain_ids[index] for index,id in enumerate(reference_chain_ids)}