import pickle
import numpy as np
import re
from absl import logging

XL_DISTRI = {
    'DSS': np.array(
        [0.        , 0.00614887, 0.0092233 , 0.0368932 , 0.00614887,
        0.02459547, 0.04304207, 0.06148867, 0.05841424, 0.07993528,
        0.09838188, 0.05841424, 0.08300971, 0.0461165 , 0.04919094,
        0.0461165 , 0.05841424, 0.0368932 , 0.03996764, 0.02152104,
        0.01537217, 0.01537217, 0.01537217, 0.00307443, 0.01537217,
        0.00614887, 0.01537217, 0.01666667, 0.01666667, 0.01666667])
}

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
    # {1: 'ACGT', 2: 'CGTA'}
    with open(fasta_file, 'r') as f:
        fasta_dict = {}
        seq = ''
        desc = 1
        for line in f.readlines():
            if line.startswith('>'):
                if seq:
                    fasta_dict[desc] = seq
                    desc += 1
                seq = ''
                # desc = line[1:].strip()
                # assert desc not in fasta_dict, f'Duplicate chain description {desc} in fasta file'

            else:
                seq += line.strip()
        if seq:
            fasta_dict[desc] = seq
    return fasta_dict

def get_asym_id(fasta_dict):
    # this function should return the asym_id of the fasta_dict
    ids = [np.repeat(i+1, len(seq)) for i, seq in enumerate(fasta_dict.values())]
    return np.concatenate(ids)


def get_mapping(fasta_dict):
    abs_to_rel = {} # abs_pos_idx: f'{chain_id}-{res_id}-{res_type}'
    res_to_abs = {} # (chain_id, res_id): (abs_pos_idx, res_type)
    idx = 0
    for i, seq in fasta_dict.items():
        for j, res_type in enumerate(seq, 1):
            abs_to_rel[idx] = f'{i}-{j}-{res_type}'
            res_to_abs[(i, j)] = (idx, res_type)
            idx += 1
    return abs_to_rel, res_to_abs


class Restraints:
    def __init__(self, fasta_file):
        fasta_dict = get_fasta_dict(fasta_file)
        fasta_dict = reorder_seq_dict(fasta_dict)
        self.abs_to_rel, self.rel_to_abs = get_mapping(fasta_dict)
        self.seqlen = sum(len(seq) for seq in fasta_dict.values())

    def load_restraints(self, restr_file):
        if restr_file.endswith('.pkl'):
            with open(restr_file, 'rb') as f:
                restr_dict = pickle.load(f)
            restr_dict = {k:v for k,v in restr_dict.items() if k in ['sbr', 'sbr_mask', 'interface_mask']}
            return restr_dict
        else:
            with open(restr_file, 'r') as f:
                contents = [i.strip() for i in f.readlines()]
            contents = [i.replace(' ', '') for i in contents]
            contents = [re.sub(r',0.050*$', '', i) for i in contents]
            return contents
            
    def convert_restraints(self, restr, out_file=None, debug=False):
        if isinstance(restr, str):
            restr = self.load_restraints(restr)
        if isinstance(restr, list):
            restr1 = self.textlist_to_arraydict(restr)
            if out_file:
                with open(out_file, 'wb') as f:
                    pickle.dump(restr1, f)
                logging.info(f'Restraints saved to {out_file}')
        else:
            restr1 = self.arraydict_to_textlist(restr)
            if out_file:
                with open(out_file, 'w') as f:
                    f.write('\n'.join(restr1)+'\n')
                logging.info(f'Restraints saved to {out_file}')
        if debug:
            restr2 = self.convert_restraints(restr1, debug=False)
            self.compare_restraints(restr, restr2)

        return restr1

    def compare_restraints(self, r1, r2):
        if isinstance(r1, list):
            assert len(r1)==len(r2), f'Length of restraints lists are different: {len(r1)} vs {len(r2)}'
            r1.sort()
            r2.sort()
            for i, j in zip(r1, r2):
                assert i==j, f'Restraints are different: {i} vs {j}'
        else:
            for k in r1:
                assert np.allclose(r1[k], r2[k]), f'Restraints are different for {k}'


    def arraydict_to_textlist(self, restr_dict):
        # this function should return a list of strings, each string contains one restraint
        # fasta_dict = self.fasta_dict
        # my_dict = {} # index: f'{chain_id}-{res_id}-{res_type}'
        # idx = 0
        # for i, seq in enumerate(fasta_dict.values(), 1):
        #     for j, res_type in enumerate(seq, 1):
        #         my_dict[idx] = f'{i}-{j}-{res_type}'
        #         idx += 1
        
        def get_cutoff(distri):
            for k, v in XL_DISTRI.items():
                if np.allclose(v, distri):
                    return k
            high_bin_num = int((distri > 1/len(distri)).sum())
            cutoff = BINS[high_bin_num-1]
            fdr = distri[high_bin_num:].sum()
            if np.abs(fdr-0.05)<0.001:
                return f'{cutoff}'
            return f'{cutoff},{round(fdr, 3)}'
        
        # RPR
        rprs = []
        for i, j in np.argwhere(restr_dict['sbr_mask']):
            
            if i>j:
                continue
            distri = restr_dict['sbr'][i, j]
            cutoff = get_cutoff(distri)
            rel_pos1 = self.abs_to_rel[i]
            rel_pos2 = self.abs_to_rel[j]
            rprs.append(f'{rel_pos1},{rel_pos2},{cutoff}')
        # IR
        irs = []
        for i in np.where(restr_dict['interface_mask'])[0]:
            rel_pos = self.abs_to_rel[i]
            irs.append(rel_pos)
        logging.info(f'Total number of restraints: {len(irs)} IRs and {len(rprs)} RPRs')
        return irs+rprs
    
    def textlist_to_arraydict(self, restr_list):
        # this function should return a dictionary that maps the restraint types to their corresponding arrays
        def get_site_pos(x):
            chain_id, res_id, res_type = x.split('-')
            abs_pos, res_type_in_fasta = self.rel_to_abs[(int(chain_id), int(res_id))]
            assert res_type_in_fasta == res_type, f'Line {i+1}: Residue type {res_type} in restraint file at position {res_id} in chain No.{chain_id} does not match fasta {res_type_in_fasta}'
            return abs_pos
        
        def get_distri(cutoff, fdr=0.05):
            if cutoff in XL_DISTRI:
                logging.info(f'Using pre-defined XL distribution for {cutoff}, fdr argument will be ignored')
                return XL_DISTRI[cutoff]
            cutoff = float(cutoff)
            xbool = np.concatenate([BINS, [np.inf]])<=cutoff
            x = np.ones(len(BINS)+1)
            x[xbool] = (1-fdr) * (x[xbool]/x[xbool].sum())
            x[~xbool] = fdr * (x[~xbool]/x[~xbool].sum())
            assert x[xbool].max() > x[~xbool].max(), (x[xbool].max(), x[~xbool].max())
            return x

        # initialize the restraints dictionary
        tot_len = self.seqlen
        restraints = {
            'interface_mask': np.zeros(tot_len),
            'sbr_mask': np.zeros((tot_len, tot_len)),
            'sbr': np.zeros((tot_len, tot_len, len(BINS)+1)),
        }
        
        for i, line in enumerate(restr_list):
            logging.info(line)
            xs = [i.strip() for i in line.split(',')]
            if len(xs) == 1:
                restraints['interface_mask'][get_site_pos(xs[0])] = 1
            elif len(xs) == 3 or len(xs) == 4:
                pos1 = get_site_pos(xs[0])
                pos2 = get_site_pos(xs[1])
                cutoff = xs[2]
                if len(xs) == 4:
                    fdr = float(xs[3])
                    distri = get_distri(cutoff, fdr=fdr)
                else:
                    distri = get_distri(cutoff)
                assert restraints['sbr_mask'][pos1, pos2] == 0, f'Line {i+1}: Restraint already exists between {xs[0]} and {xs[1]}'
                assert restraints['sbr_mask'][pos2, pos1] == 0, f'Line {i+1}: Restraint already exists between {xs[1]} and {xs[0]}'
                restraints['sbr_mask'][pos1, pos2] = 1
                restraints['sbr_mask'][pos2, pos1] = 1
                restraints['sbr'][pos1, pos2] = distri
                restraints['sbr'][pos2, pos1] = distri
            else:
                raise ValueError(f'Line {i+1}: Invalid restraint format')
        logging.info(f'Total number of restraints: {int(restraints["interface_mask"].sum())} IRs and {int(restraints["sbr_mask"].sum())//2} RPRs')
        return restraints


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Convert restraints from text (txt file) to array dictionary (pkl file) format and vice versa')
    parser.add_argument('restraints_file', type=str, help='txt or pkl file containing the restraints. See README for the format.')
    parser.add_argument('fasta_file', type=str, help='path to the fasta file')
    parser.add_argument('-o', '--output_file', type=str, default=None, help='path to the output file. If not specified, the output file will be the same as the restraints file with extension changed to pkl or txt, depending on the input file format.')
    parser.add_argument('-d', '--debug', action='store_true', help='run additional checks to verify the conversion')
    args = parser.parse_args()
    logging.set_verbosity(logging.INFO)
    

    r = Restraints(args.fasta_file)
    if args.output_file is None:
        if args.restraints_file.endswith('.pkl'):
            output_file = args.restraints_file.replace('.pkl', '')+'.txt'
        else:
            output_file = args.restraints_file.replace('.txt', '')+'.pkl'
    else:
        output_file = args.output_file
    
    restr = r.convert_restraints(args.restraints_file, out_file=output_file, debug=args.debug)