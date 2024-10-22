import pickle
import numpy as np
import re


BINS = np.arange(4, 33, 1)
def reorder_seq_dict(seq_dict):
    # this function should return a new dictionary that maps the chain ids to the sequences in the order of the restraints dict,
    # where same sequences are grouped together
    # for example, if the restraints dict is {'A': 'ACGT', 'B': 'CGTA', 'C': 'ACGT', 'D': 'CGTA'}, the function should return:
    # {'A': 'ACGT', 'C': 'ACGT', 'B': 'CGTA', 'D': 'CGTA'}
    
    # get all unique values in the sequence dictionary
    unique_values_order = []
    for value in seq_dict.values():
        if value not in unique_values_order:
            unique_values_order.append(value)
    
    # create a new dictionary with the same order as the restraints dict
    reordered_dict = {k: v for v in unique_values_order for k in seq_dict if seq_dict[k] == v}
    
    return reordered_dict


def get_fasta_dict(fasta_file):
    # this function should return a dictionary that maps fasta chain ids to its sequence, 
    # for example, if the fasta file contains two sequences, the dictionary should be:
    # {'A': 'ACGT', 'B: 'CGTA'}
    with open(fasta_file, 'r') as f:
        fasta_dict = {}
        seq = ''
        for line in f.readlines():
            if line.startswith('>'):
                if seq:
                    fasta_dict[desc] = seq
                seq = ''
                desc = line[1:].strip()
                assert desc not in fasta_dict, f'Duplicate chain description {desc} in fasta file'
            else:
                seq += line.strip()
        if seq:
            fasta_dict[desc] = seq
    return fasta_dict

def get_asym_id(fasta_dict):
    # this function should return the asym_id of the fasta_dict
    ids = [np.repeat(i+1, len(seq)) for i, seq in enumerate(fasta_dict.values())]
    return np.concatenate(ids)


def get_restr_dict(restraints_file, fasta_file):
    # each line contains one restraint, with the format:
    # "chainindex1-residueindex1-residuetype1, chainindex1-residueindex1-residuetype1, distance_cutoff" for the RPR restraint
    # "chainindex1-residueindex1-residuetype1" for the IR restraint

    fasta_dict = get_fasta_dict(fasta_file)
    fasta_dict = {i+1: v for i, v in enumerate(fasta_dict.values())}
    fasta_dict = reorder_seq_dict(fasta_dict)
    asym_id = get_asym_id(fasta_dict)
    seqlens = [len(v) for v in fasta_dict.values()]
    cum_seqlen_dict = {k: sum(seqlens[:list(fasta_dict.keys()).index(k)]) for k in fasta_dict.keys()}
    allseqs = ''.join(fasta_dict.values())
    with open(restraints_file, 'r') as f:
        contents = [i.strip() for i in f.readlines()]
    
    def get_site_pos(x):
        chain_id, res_id, res_type = x.split('-')
        chain_id = int(chain_id)
        res_id = int(res_id)
        assert fasta_dict[chain_id][res_id-1] == res_type, f'Line {i+1}: Residue type {res_type} at position {res_id} in chain No.{chain_id} does not match sequence {fasta_dict[chain_id][res_id-1]}'
        pos = cum_seqlen_dict[chain_id]+res_id-1
        assert allseqs[pos] == res_type, f'Line {i+1}: Residue type {res_type} does not match total sequence at position {pos} {allseqs[pos]}'
        return pos
    
    def get_distri(cutoff, fdr=0.05):
        xbool = np.concatenate([BINS, [np.inf]])<=cutoff
        x = np.ones(len(BINS)+1)
        x[xbool] = (1-fdr) * (x[xbool]/x[xbool].sum())
        x[~xbool] = fdr * (x[~xbool]/x[~xbool].sum())
        assert x[xbool].max() > x[~xbool].max(), (x[xbool].max(), x[~xbool].max())
        return x

    # initialize the restraints dictionary
    tot_len = sum(seqlens)
    restraints = {
        'interface_mask': np.zeros(tot_len),
        'sbr_mask': np.zeros((tot_len, tot_len)),
        'sbr': np.zeros((tot_len, tot_len, len(BINS)+1)),
        'asym_id': asym_id, # for debugging purposes
    }
    
    for i, line in enumerate(contents):
        xs = [i.strip() for i in line.split(',')]
        if len(xs) == 1:
            restraints['interface_mask'][get_site_pos(xs[0])] = 1
        elif len(xs) == 3:
            pos1 = get_site_pos(xs[0])
            pos2 = get_site_pos(xs[1])
            cutoff = float(xs[2])
            distri = get_distri(cutoff)
            restraints['sbr_mask'][pos1, pos2] = 1
            restraints['sbr'][pos1, pos2] = distri
        else:
            raise ValueError(f'Line {i+1}: Invalid restraint format')
    return restraints


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('restraints_file', type=str, help='path to the restraints file. Each line contains one restraint, with the format: "chainindex1-residueindex1-residuetype1, chainindex1-residueindex1-residuetype1, distance_cutoff" for the RPR restraint, "chainindex1-residueindex1-residuetype1" for the IR restraint.')
    parser.add_argument('fasta_file', type=str, help='path to the fasta file')
    parser.add_argument('--output_file', type=str, default=None, help='path to the output file. If not specified, the output file will be the same as the restraints file with a .pkl extension.')
    args = parser.parse_args()

    restr_dict = my_get_restr_dict(args.restraints_file, args.fasta_file)
    if args.output_file is None:
        output_file = args.restraints_file.replace('.txt', '.pkl')
    else:
        output_file = args.output_file
    with open(output_file, 'wb') as f:
        pickle.dump(restr_dict, f)