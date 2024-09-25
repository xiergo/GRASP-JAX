import numpy as np 
from alphafold.common.residue_constants import atom_order,restype_order

from alphafold.common.protein import PDB_CHAIN_IDS
import logging
def dist_onehot(dist, bins):
    x = (dist[..., None] > bins).sum(-1)
    return np.eye(len(bins) + 1)[x]

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """compute pseudo beta features from atom positions"""
    is_gly = np.equal(aatype, restype_order['G'])
    ca_idx = atom_order['CA']
    cb_idx = atom_order['CB']
    pseudo_beta = np.where(
        np.tile(is_gly[..., None].astype("int32"), [1] * len(is_gly.shape) + [3]).astype("bool"),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])
    if all_atom_masks is not None:
        pseudo_beta_mask = np.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
        return pseudo_beta, pseudo_beta_mask
    return pseudo_beta

def get_dist_from_protein(prot):
    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(prot.aatype, prot.atom_positions, prot.atom_mask)
    pred_dist = np.sqrt(((pseudo_beta[:, None] - pseudo_beta[None]) ** 2).sum(-1) + 1e-8)
    pseudo_beta_mask_2d = pseudo_beta_mask[:, None] * pseudo_beta_mask[None]
    return pred_dist, pseudo_beta_mask_2d
def get_nbdist_avg_ca(prot, asym_id, break_thre=5.0):
    """compute averaged neihbour ca distance for each residue"""
    # atom_types = [
    #     'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    #     'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    #     'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    #     'CZ3', 'NZ', 'OXT'
    # ]
    ca_idx = 1
    ca_pos = prot.atom_positions[..., ca_idx, :] #[nres, natom, 3]
    nbdist = np.sqrt(((ca_pos[1:]-ca_pos[:-1])**2).sum(-1)+1e-8)
    nbdist_leftadd = np.concatenate([[nbdist[0]], nbdist])
    nbdist_rightadd = np.concatenate([nbdist, [nbdist[-1]]])
    is_chain_start = asym_id!=np.concatenate(([[-1], asym_id[:-1]]))
    is_chain_end = asym_id!=np.concatenate((asym_id[1:], [100000]))
    nbdist_left = np.where(is_chain_start, nbdist_rightadd, nbdist_leftadd)
    nbdist_right = np.where(is_chain_end, nbdist_leftadd, nbdist_rightadd)
    nbdist_avg = (nbdist_left+nbdist_right)/2

    break_num = int((nbdist_left>break_thre).sum())
    max_nb_dist = nbdist_left.max()

    return nbdist_avg, break_num, max_nb_dist

def compute_recall(satis, mask, conf):
    if mask.sum() <= 0:
        return None, None
    
    recall = (satis*mask).sum()/(mask.sum()+1e-8)
    recall_conf = (satis*mask*conf).sum()/((mask*conf).sum())
    return recall, recall_conf

def compute_rm_score(values, thres):
    score = 0
    scale = 1
    assert len(values) == len(thres), (values, thres)
    for value, thre in zip(values, thres):
        if (thre is not None):
            if (value>thre):
                score += (value-thre)*scale
            scale *= 100
    return score

def get_range(x):
    BINS = np.arange(4, 33, 1)
    lowers = np.concatenate([[0], BINS])
    uppers = np.concatenate([BINS, [np.inf]])
    intervals = [(i, j) for i, j, k in zip(lowers, uppers, x) if k]
    ys = []
    last = None
    for i, j in intervals:
        if (last is not None) and (last == i):
            ys[-1][-1] = j
        else:
            ys.append([i, j])
        last = j
    return ','.join([f'{i}-{j}' for i, j in ys])

def generate_terminal_mask(asym_id, n):
    is_end = asym_id != np.concatenate([asym_id[1:], [-1]])
    is_start = asym_id != np.concatenate([[0], asym_id[: -1]])
    end_idx = np.where(is_end)[0]
    start_idx = np.where(is_start)[0]
    term_idx = np.concatenate([end_idx, start_idx])
    idx = np.arange(len(asym_id))
    mask = (np.abs(idx[:, None] - term_idx[None]) >= n).all(axis=-1).astype(int)
    mask = mask[None] # only mask the other side which is different from the interface.
    return mask

def filter_restraints(restraints, restraints0, prot, update = True, nbdist_ca_thre=5, max_rm_ratio=0.2, viol_thre=5, mask_terminal_residues=2):
    # restraints0: initial restraints.
    # restraints: current restraints.
    BINS = np.arange(4, 33, 1)
    plddt = prot.b_factors.max(-1)
    pred_dist, pseudo_beta_mask_2d = get_dist_from_protein(prot)
    mask_intrachain = restraints['asym_id'][None] == restraints['asym_id'][:, None]
    terminal_residue_mask = generate_terminal_mask(restraints['asym_id'], mask_terminal_residues)
    d = pred_dist + mask_intrachain*1000 + (1-pseudo_beta_mask_2d) * 1000 + (1-terminal_residue_mask) * 1000
    # dist_thre=10.0
    # plddts_2d = (d<=dist_thre)*plddt[None]
    # plddt_otherside = plddts_2d.max(axis=1)
    sbr = restraints['sbr']
    sbr_high = (sbr > (1 / sbr.shape[-1]))

    not_high_bin = 1-sbr_high
    upper_1d = np.concatenate([BINS, [100,]])
    sbr_upper_thre = (upper_1d-1e6*not_high_bin).max(-1)
    sbr_upper_viol_dist = (pred_dist-sbr_upper_thre)
    sbr_max_viol_dist = (sbr_upper_viol_dist * restraints['sbr_mask']).max()
    sbr_viol_num = ((sbr_upper_viol_dist * restraints['sbr_mask']) > 0).sum() / 2
    interface_viol_dist = ((d.min(axis=-1)-8.0)*restraints['interface_mask'])
    interface_max_viol_dist = interface_viol_dist.max()
    interface_viol_num = (interface_viol_dist>0).sum()
    viol_num = sbr_viol_num + interface_viol_num
    max_viol_dist = max(sbr_max_viol_dist, interface_max_viol_dist)
    pred_dist_onehot = dist_onehot(pred_dist, BINS)
    sbr_satis = (sbr_high * pred_dist_onehot).sum(-1) * pseudo_beta_mask_2d
    nbdist_avg_ca, break_num, max_nb_dist = get_nbdist_avg_ca(prot, asym_id=restraints['asym_id'])
    includ_mat = np.zeros_like(restraints['sbr_mask'])
    includ_if = np.zeros_like(restraints['interface_mask'])
    
    
    def resi(i, ds=None):
        cid = PDB_CHAIN_IDS[int(restraints['asym_id'][i])-1]
        rid = prot.residue_index[i]
        y = f'{cid}{rid}/conf{plddt[i]:.2f}/nbdist_avg_ca{nbdist_avg_ca[i]:.2f}'
        if ds is not None:
            y += f'/dist_cb{ds[i]:.2f}'
        return y
    

    def print_pair(ps):
        ps = [(i, j) for i, j in ps if i<j]
        satisfied_num = 0
        included_num = 0

        nbdists = [(nbdist_avg_ca[i]+nbdist_avg_ca[j])/2 for i, j in ps]
        viol_dists = [sbr_upper_viol_dist[i, j] for i, j in ps]
        rm_scores = [compute_rm_score((viol_dist, nb_dist), (viol_thre, nbdist_ca_thre)) for nb_dist, viol_dist in zip(nbdists, viol_dists)]
        rm_thre = np.quantile(rm_scores, 1-max_rm_ratio)
        
        for (i, j), rm_score in zip(ps, rm_scores):
            if (rm_score <= rm_thre):
                includ_mat[i,j] = 1
                includ_mat[j,i] = 1
                included_num += 1
                filter_info = 'Included!'
            else:
                filter_info = 'Excluded!'
            if sbr_satis[i,j]:
                satisfied_num += 1
                satis_info = 'Satisfied!'
            else:
                satis_info = 'Violated! '
            logging.info(f'{filter_info} {satis_info} {resi(i)}<==>{resi(j, pred_dist[i])}, range: {get_range(sbr_high[i,j])}, rm_score {rm_score}, rm_thre {rm_thre}')
        logging.info(f'>>>>> Total {len(ps)}: {included_num} included, {satisfied_num} satisfied')
    
    # print interface info ==========================================================
    if_num = int(restraints['interface_mask'].sum())
    if if_num>0:
        logging.info('interface restraints:')
        included_num = 0
        satisfied_num = 0
        nbdists = [nbdist_avg_ca[i] for i in np.where(restraints['interface_mask'])[0]]
        viol_dists = [d[i].min()-8.0 for i in np.where(restraints['interface_mask'])[0]]
        rm_scores = [compute_rm_score((viol_dist, nb_dist), (viol_thre, nbdist_ca_thre)) for nb_dist, viol_dist in zip(nbdists, viol_dists)]
        rm_thre = np.quantile(rm_scores, 1-max_rm_ratio)
        for i, rm_score in zip(np.where(restraints['interface_mask'])[0], rm_scores):
            # js = np.where((plddts_2d[i])>0)[0]
            if d[i].min()<=8.0:
                satisfied_num += 1
                satis_info = 'Satisfied!'
            else:
                satis_info = 'Violated! '
                
            # if len(js)==0:
            #     logging.info(f'Excluded! {satis_info} {resi(i)}<==>{resi(np.argmin(ds), ds)}')
            # else:
                # jmax = np.argmax(plddts_2d[i])
            
            if (rm_score<=rm_thre):
                includ_if[i] = 1
                included_num += 1
                filter_info = 'Included!'
            else:
                filter_info = 'Excluded!'
            logging.info(f'{filter_info} {satis_info} {resi(i)} {d[i].min()}, rm_score{rm_score}, rm_thre{rm_thre}')

        logging.info(f'>>>>> Total {if_num}, {included_num} included, {satisfied_num} satisfied')
    
    # print sbr info =================================================================
    intra_ps = np.transpose(np.where(restraints['sbr_mask']*mask_intrachain))
    inter_ps = np.transpose(np.where(restraints['sbr_mask']*(1-mask_intrachain)))
    intra_sbr = int(len(intra_ps)/2)
    inter_sbr = int(len(inter_ps)/2)
    tot_sbr = intra_sbr+inter_sbr
    if tot_sbr >0:          
        logging.info(f'inter-residue restraints: {tot_sbr}({inter_sbr} inter-chain + {intra_sbr} intra-chain)')
    if inter_sbr > 0:
        logging.info('Inter-chain restraints')
        print_pair(inter_ps)
    if intra_sbr > 0:
        logging.info('Intra-chain restraints')
        print_pair(intra_ps)
    
    # update restraints based on plddts ==============================================
    tot_before = int(tot_sbr+if_num)
    if update == False:
        includ_if = np.ones_like(restraints['interface_mask'])
        includ_mat = np.ones_like(restraints['sbr_mask'])
    restraints['interface_mask'] = includ_if * restraints['interface_mask']
    restraints['sbr_mask'] = includ_mat * restraints['sbr_mask']
    restraints['sbr'] = restraints['sbr'] * restraints['sbr_mask'][:,:,None]
    tot_after = int((restraints['interface_mask']).sum() + (restraints['sbr_mask']).sum()/2)
    rm_num = int(tot_before - tot_after)

    # compute recall, breakage
    sbr_mask0 = restraints0['sbr_mask']
    sbr0 = restraints0['sbr']
    sbr_high0 = (sbr0 > (1 / sbr0.shape[-1]))
    sbr_satis0 = (sbr_high0 * pred_dist_onehot).sum(-1) * pseudo_beta_mask_2d


    
    
    interface_mask0 = restraints0['interface_mask']
    interface_satis0 = d.min(axis=1)<=8
    conf_2d = (plddt[None]+plddt[:, None])/2

    recall_dict = {
        'interchain': (*compute_recall(sbr_satis0, sbr_mask0*np.triu(sbr_mask0)*(1-mask_intrachain), conf_2d), 1),
        'intrachain': (*compute_recall(sbr_satis0, sbr_mask0*np.triu(sbr_mask0)*mask_intrachain, conf_2d), 0.5),
        'interface':  (*compute_recall(interface_satis0, interface_mask0, plddt), 1)
    }
    
    recall_dict = {
        k: v for k, v in recall_dict.items() if v[0] is not None
    }


    logging.info('Breakage info ==========')
    logging.info(f'Break number: {break_num}, Max neigbour CA dist: {max_nb_dist}\n')

    logging.info('Recall info=============')
    recall = 0
    recall_conf = 0
    w = 0

    for k, v in recall_dict.items():
        if v[0] is None:
            continue
        logging.info(f'{k} (w {v[2]}): recall {v[0]}, recall weighted by confidence: {v[1]}')
        recall += v[0]*v[2]
        recall_conf += v[1]*v[2]
        w += v[2]

    if w == 0:
        # no restraints
        recall = None
        recall_conf = None
    else:
        recall /= w
        recall_conf /= w

    return rm_num, break_num, max_nb_dist, recall, recall_conf, viol_num, max_viol_dist
