from Bio import PDB
import copy
import sys
import os
import pdb

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
    chains.sort()
    print(chains)
    grouped = []
    while chains:
        chain = chains.pop(0)
        group = [chain]
        for other_chain in chains:
            if compare_chains(chain, other_chain):
                group.append(other_chain)
                chains.remove(other_chain)
        grouped.append(group)
    return grouped

def reorder_chains(structure, name, grouped_chains, output_dir):
    """Reorder chains to make identical chains adjacent."""
    chain_id = 'A'
    count_num = 1
    for group in grouped_chains:
        folder_name = f'{output_dir}/{name}_{count_num}'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for chain in group:
            if chain.id > chain_id:
                chain.id = chain_id                        
            output_file = f"{folder_name}/{name}_{chain_id}.pdb"
            io = PDB.PDBIO()
            io.set_structure(chain)
            io.save(output_file)
            chain_id = chr(ord(chain_id) + 1)
        count_num += 1
            
def main(input_file, name, output_dir):
    parser = PDB.PDBParser()
    structure = parser.get_structure(name, input_file)
    grouped_chains = group_chains(structure)
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    reorder_chains(structure, name, grouped_chains, output_dir)

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    output_dir = os.path.abspath(output_dir)
    name = sys.argv[3]
    main(input_file, name, output_dir)