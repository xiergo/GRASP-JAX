# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script."""
import enum
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Any, Dict, Union
import pandas as pd
from absl import app
from absl import flags
from absl import logging
from alphafold.common import confidence
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.common.restraints_process import filter_restraints
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax
from alphafold.data.tools.align import interpolate_structures,extract_restraints,renumber_chains,get_chain_sequences,reorder_and_renumber_chains,\
restore_chains_from_merge,parse_pdb_chains,random_group_by_proximity,merge_chains_by_group,chain_mapping
import jax.numpy as jnp
import numpy as np
import tree
import subprocess
import copy
# Internal import (7716).
from alphafold.data.tools import combfit_newest
from alphafold.data.tools import parse_template
# logging.set_verbosity(logging.INFO)


@enum.unique
class ModelsToRelax(enum.Enum):
  ALL = 0
  BEST = 1
  NONE = 2

flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')
flags.DEFINE_string('mrc_path', None, 'Path to the mrc file')
flags.DEFINE_float('resolution', 10, 'Resolution of the mrc file')
flags.DEFINE_string('name', None, 'Name of the protein')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('feature_pickle', None, 'Path to a precomputed_feature dict')
flags.DEFINE_string('restraints_pickle', None, 'Path to a precomputed_feature dict')
flags.DEFINE_string('jackhmmer_binary_path', shutil.which('jackhmmer'),
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', shutil.which('hhblits'),
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', shutil.which('hhsearch'),
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', shutil.which('hmmsearch'),
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', shutil.which('hmmbuild'),
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('kalign_binary_path', shutil.which('kalign'),
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', None, 'Path to the small '
                    'version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniref30_database_path', None, 'Path to the UniRef30 '
                    'database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', None, 'Path to the Uniprot '
                    'database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'Path to the PDB '
                    'seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_enum('db_preset', 'full_dbs',
                  ['full_dbs', 'reduced_dbs'],
                  'Choose preset MSA database configuration - '
                  'smaller genetic database config (reduced_dbs) or '
                  'full genetic database config  (full_dbs)')
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer','multimer-1','multimer-2','multimer-3','multimer-4','multimer-5'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('num_multimer_predictions_per_model', 5, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. '
                     'Note: this FLAG only applies if model_preset=multimer')
flags.DEFINE_boolean('use_precomputed_msas', False, 'Whether to read MSAs that '
                     'have been written to disk instead of running the MSA '
                     'tools. The MSA files are looked up in the output '
                     'directory, so it must stay the same between multiple '
                     'runs that are to reuse the MSAs. WARNING: This will not '
                     'check if the sequence, database or configuration have '
                     'changed.')
flags.DEFINE_enum_class('models_to_relax', ModelsToRelax.BEST, ModelsToRelax,
                        'The models to run the final relaxation step on. '
                        'If `all`, all models are relaxed, which may be time '
                        'consuming. If `best`, only the most confident model '
                        'is relaxed. If `none`, relaxation is not run. Turning '
                        'off relaxation might result in predictions with '
                        'distracting stereochemical violations but might help '
                        'in case you are having issues with the relaxation '
                        'stage.')
flags.DEFINE_boolean('use_gpu_relax', None, 'Whether to relax on GPU. '
                     'Relax on GPU can be much faster than CPU, so it is '
                     'recommended to enable if possible. GPUs must be available'
                     ' if this setting is enabled.')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')


def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
  """Recursively changes jax arrays to numpy arrays."""
  for k, v in output.items():
    if isinstance(v, dict):
      output[k] = _jnp_to_np(v)
    elif isinstance(v, jnp.ndarray):
      output[k] = np.array(v)
  return output


def _save_confidence_json_file(
    plddt: np.ndarray, output_dir: str, model_name: str
) -> None:
  confidence_json = confidence.confidence_json(plddt)

  # Save the confidence json.
  confidence_json_output_path = os.path.join(
      output_dir, f'confidence_{model_name}.json'
  )
  with open(confidence_json_output_path, 'w') as f:
    f.write(confidence_json)


def _save_mmcif_file(
    prot: protein.Protein,
    output_dir: str,
    model_name: str,
    file_id: str,
    model_type: str,
) -> None:
  """Crate mmCIF string and save to a file.

  Args:
    prot: Protein object.
    output_dir: Directory to which files are saved.
    model_name: Name of a model.
    file_id: The file ID (usually the PDB ID) to be used in the mmCIF.
    model_type: Monomer or multimer.
  """

  mmcif_string = protein.to_mmcif(prot, file_id, model_type)

  # Save the MMCIF.
  mmcif_output_path = os.path.join(output_dir, f'{model_name}.cif')
  with open(mmcif_output_path, 'w') as f:
    f.write(mmcif_string)


def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    feature_path:str,
    restraints_path:str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int,
    models_to_relax: ModelsToRelax,
    model_type: str,
    mrc_path: str,
    resolution: float,
    name: str,
):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  # output_dir = os.path.join(output_dir_base, fasta_name)
  output_dir = output_dir_base
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Get features.
  t_0 = time.time()
  if feature_path == "None":
    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir)
    features_output_path = os.path.join(output_dir, 'features.pkl')
    with open(features_output_path, 'wb') as f:
      pickle.dump(feature_dict, f, protocol=4)
  else:
    feature_dict = pickle.load(open(feature_path,'rb'))
  # restraints = pickle.load(open('../5JDS/5JDS_restr.pkl','rb'))
  # 
  timings['features'] = time.time() - t_0

  # Write out features as a pickled dictionary.
  if restraints_path == "None":
    BINS = np.arange(4, 33, 1)
    dtype = np.float32
    seq_length = len(feature_dict['aatype'])
    restraints = {'sbr': np.zeros((seq_length, seq_length, len(BINS)+1)).astype(dtype),
      'sbr_mask' : np.zeros((seq_length, seq_length)).astype(dtype),
      'interface_mask': np.zeros(seq_length).astype(dtype)}
    logging.info('empty restraints ')
  else:
    restraints = pickle.load(open(restraints_path,'rb'))
    logging.info('read restraints successfully')

  feature_dict.update(restraints)
  logging.info('Using quick inference')
  is_complex = True
  unrelaxed_pdbs = {}
  unrelaxed_proteins = {}
  ranking_confidences = {}
  nbdist_ca_thre=5.0
  viol_thre=5.0
  heter = False
  # # iter=5
  max_rm_ratio=0.2
  mask_terminal_residues=0
  left_ratio=0.2
  restraints0 = {
        'sbr': feature_dict['sbr'],
        'sbr_mask': feature_dict['sbr_mask'],
        'interface_mask': feature_dict['interface_mask'],
        'asym_id': feature_dict['asym_id']
    }
  left_thre = (restraints0['interface_mask'].sum() + restraints0['sbr_mask'].sum()/2)*left_ratio
  left_thre = int(np.ceil(left_thre))
  logging.info('At least %d restraints will be used in the final iteration', left_thre)
  # Run the models.
  align_density_path = None
  logging.info('mrc_path: %s', mrc_path)
  if mrc_path != "None":
    align_density_path = os.path.join(output_dir,'align_density')
    if not os.path.exists(align_density_path):
      os.mkdir(align_density_path)
    # if np.sum(feature_dict['entity_id'] == 1) == len(feature_dict['entity_id']):
    #   homomer_chains = feature_dict['asym_id'][-1]
    #   combfitter = Combfit_homomer(mrc_path = mrc_path, resolution = resolution, name = name, homomer_chains=homomer_chains)
    # elif feature_dict['entity_id'][-1] < feature_dict['asym_id'][-1]:
    #   heter = True
    #   # pdb_paths = heteroligomer_list_construct(args.out)
    #   combfitter = Combfit_heteoligomer(mrc_path = mrc_path, resolution = resolution, name = name)
    #   # combfitter.docking_pipeline_heteoligomer()
    # else:
    #   combfitter = Combfit(mrc_path = mrc_path, resolution = resolution, name = name)
  num_models = len(model_runners)
  transform_em = 'restraints'
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()):
    logging.info('Running model %s on %s', model_name, fasta_name)
    feature_dict.update(restraints0)
    # t_0 = time.time()
    restraints = copy.deepcopy(restraints0)
    mydicts = []
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed)
    # timings[f'process_features_{model_name}'] = time.time() - t_0
    t_0 = time.time()
    # feat0 = copy.deepcopy(processed_feature_dict)
    
    def callback(result, restraints, recycles,feat,it, num_recycle_cur_iter, prev):
      if recycles == 0: 
        result.pop("tol",None)
      print_line = ""
      for x,y in [["mean_plddt","pLDDT"],["ptm","pTM"],["iptm","ipTM"],["tol","tol"]]:
        if x in result:
          print_line += f" {y}={result[x]:.3g}"
      logging.info("%s recycle=%s iter=%s local_iter=%s%s",model_name,recycles,it,num_recycle_cur_iter,print_line)
      num_recycle_cur_iter += 1
      is_continue = True
      # if (recycles > 0 and result["tol"] < 0.5) or num_recycle_cur_iter >= 4:
      #     final_atom_mask = result["structure_module"]["final_atom_mask"]
      #     b_factors = result["plddt"][:, None] * final_atom_mask
      #     confidence = result["mean_plddt"]
      #     unrelaxed_protein = protein.from_prediction(
      #         features=feat,
      #         result=result, b_factors=b_factors,
      #         remove_leading_feature_dimension=not model_runner.multimer_mode)
      #     unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}_{it+1}.pdb')
      #     with open(unrelaxed_pdb_path, 'w') as f:
      #         f.write(protein.to_pdb(unrelaxed_protein))
      #     restraints = {i:feat[i] for i in ['sbr','sbr_mask','interface_mask','asym_id']}
      #     ranking_score = result['ranking_confidence']
      #     rm_num, break_num, max_nb_dist, recall, recall_conf, viol_num, max_viol_dist = filter_restraints(restraints, restraints0, unrelaxed_protein, nbdist_ca_thre=nbdist_ca_thre, max_rm_ratio=max_rm_ratio, viol_thre=viol_thre, mask_terminal_residues=mask_terminal_residues)
      #     rest = int(restraints['interface_mask'].sum() + restraints['sbr_mask'].sum()/2)
      #     for i in ['sbr','sbr_mask','interface_mask','asym_id']:
      #         feat[i] = restraints[i]
      #     assert rm_num >=0, rm_num
      #     mydict = {
      #       'Iter': it+1,
      #       'Conf': confidence,
      #       'RankScore': ranking_score,
      #       'Total': rm_num+rest,
      #       'Remove': rm_num,
      #       'Rest': rest,
      #       'MaxNbDist': max_nb_dist,
      #       'BreakNum': break_num,
      #       'Recall': recall,
      #       'RecallByConf': recall_conf,
      #       'Recycle_num': num_recycle_cur_iter,
      #       'Diff': result['tol'],
      #       'ViolNum': int(viol_num),
      #       'MaxViolDist': round(max_viol_dist, 2),
      #       # 'Time': round(time.time()-t_0, 2)
      #     }
      #     mydicts.append(mydict)
      #     # if (rest <= left_thre) or (rm_num == 0) or (it>=4):
      #     if it>=4:
      #         is_continue = False
      #     # t_0 = time.time()
      #     it += 1
      #     num_recycle_cur_iter = 0
      if ((recycles > 0 and result['tol'] <= 0.5) and num_recycle_cur_iter >= 15) or (num_recycle_cur_iter >= 25):
        it += 1
        # num_recycle_cur_iter = 0
        if it > 0  and align_density_path is not None:
          final_atom_mask = result["structure_module"]["final_atom_mask"]
          b_factors = result["plddt"][:, None] * final_atom_mask
          unrelaxed_protein = protein.from_prediction(
              features=feat,
              result=result, b_factors=b_factors,
              remove_leading_feature_dimension=not model_runner.multimer_mode)
          align_density_pdb_path = os.path.join(align_density_path, f'{model_name}_{it}_{num_recycle_cur_iter}.pdb')
          density_work_dir = os.path.join(align_density_path, f'{model_name}_{it}_{num_recycle_cur_iter}')
          if not os.path.exists(density_work_dir):
            os.mkdir(density_work_dir)
          # combfit_align_output = os.path.join(align_density_path, f'combfit_{model_name}_{it}_{num_recycle_cur_iter}.pdb')
          # with open(align_density_pdb_path, 'w') as f:
          #   f.write(protein.to_pdb(unrelaxed_protein))
          
          restraints = {i:feat[i] for i in ['sbr','sbr_mask','interface_mask','asym_id']}
          rm_num, break_num, max_nb_dist, recall, recall_conf, viol_num, max_viol_dist = filter_restraints(restraints, restraints, unrelaxed_protein, update=False,nbdist_ca_thre=nbdist_ca_thre, max_rm_ratio=max_rm_ratio, viol_thre=viol_thre, mask_terminal_residues=mask_terminal_residues)
          rest = int(restraints['interface_mask'].sum() + restraints['sbr_mask'].sum()/2)
          ranking_score = result['ranking_confidence']
          confidence = result["mean_plddt"]
          mydict = {
              'Iter': it,
              'Conf': confidence,
              'RankScore': ranking_score,
              'Total': rm_num+rest,
              'Remove': rm_num,
              'Rest': rest,
              'MaxNbDist': max_nb_dist,
              'BreakNum': break_num,
              'Recall': recall,
              # 'RecallByConf': recall_conf,
              'Recycle_num': num_recycle_cur_iter,
              'Diff': result['tol'],
              'ViolNum': int(viol_num),
              'MaxViolDist': round(max_viol_dist, 2),
              # 'Time': round(time.time()-t_0, 2)
            }
          mydicts.append(mydict)
          align_density_pdb_path = os.path.join(align_density_path, f'{model_name}_{it}_{num_recycle_cur_iter}.pdb')
          density_work_dir = os.path.join(align_density_path, f'{model_name}_{it}_{num_recycle_cur_iter}')
          with open(align_density_pdb_path, 'w') as f:
              f.write(protein.to_pdb(unrelaxed_protein))
          reorder_align_density_pdb_path = os.path.join(align_density_path, f'{model_name}_{it}_{num_recycle_cur_iter}_reorder.pdb')
          renumber_chains(align_density_pdb_path,reorder_align_density_pdb_path)
          #   # tmp_align_output = os.path.join(density_work_dir, f'combfit_{model_name}_{it}_{num_recycle_cur_iter}_best_structure_tmp.pdb')
          # combfit_reorder_align_output = os.path.join(align_density_path, f'combfit_{model_name}_best_structure_reorder.pdb')
          density_work_dir = os.path.abspath(density_work_dir)
          dock_initial_pdb_path = os.path.join(align_density_path, f'{model_name}_dock_initial.pdb')
          combfit_align_output = os.path.join(density_work_dir, f'model_best_structure.pdb')
          reorder_combfit_align_output = os.path.join(density_work_dir,f'combfit_{model_name}_best_structure_order.pdb')
          combfit_tmp_align_output = os.path.join(density_work_dir,f'combfit_{model_name}_best_structure_tmp.pdb')
          combfit_clean_output = os.path.join(density_work_dir,f'combfit_{model_name}_best_structure_clean.pdb')
          if it==1:
            chains = parse_pdb_chains(reorder_align_density_pdb_path)
            n_groups = np.random.randint(2, len(chains)) 
            grouped_chains = random_group_by_proximity(chains, n_groups)
            residue_mapping = merge_chains_by_group(reorder_align_density_pdb_path, grouped_chains,dock_initial_pdb_path)

            combfit_path = os.path.abspath(combfit_newest.__file__)
            base_dir= os.path.dirname(combfit_path)
            pipeline_path = os.path.join(base_dir, 'pipeline_integrate.sh')
            parse_template.main(dock_initial_pdb_path, 'model', density_work_dir)
            pdb_paths = combfit_newest.heteroligomer_list_construct(density_work_dir)
            combfitter = combfit_newest.Combfit_heteoligomer(mrc_path, resolution, density_work_dir, True, os.cpu_count(), "model", None, False, pdb_paths = pdb_paths)
            if os.path.exists(combfit_align_output):
              logging.info('Using existing results')
            else:
              t1 = time.time()
              logging.info('Running combfit')
              combfitter.docking_pipeline_heteoligomer()
              t2 = time.time()
              logging.info(f'combfit consume %ds',t2-t1)
            reference_sequences = list(get_chain_sequences(dock_initial_pdb_path).values())
            reorder_and_renumber_chains(combfit_align_output,reorder_combfit_align_output, reference_sequences)
            chain_tanvers = chain_mapping(dock_initial_pdb_path,reorder_combfit_align_output)
            residue_map = {chain_tanvers[id]:residue_mapping[id] for id in residue_mapping.keys()}
            restore_chains_from_merge(reorder_combfit_align_output,residue_map,combfit_tmp_align_output)
            reference_sequences = list(get_chain_sequences(reorder_align_density_pdb_path).values())
            reorder_and_renumber_chains(combfit_tmp_align_output,combfit_clean_output, reference_sequences)
            
            # os.system(f'python3 alphafold/data/tools/20240710_Combfit_heteoligomer.py -pdb_path {align_density_pdb_path} -mrc_path {mrc_path} -resolution {resolution} -output_path {combfit_align_output}')
            # renumber_chains(combfit_align_output,combfit_reorder_align_output)
             # reference_sequences = list(get_chain_sequences(align_density_pdb_path).values())
            # reorder_and_renumber_chains(combfit_align_output,f'./test_8cx0/group_{i}/combfit_8cx0_{i}_best_structure_reorder.pdb', reference_sequences)
            # reassign_chain_names(f'./test_8cx0/merge_{i}.pdb',f'./test_8cx0/group_{i}/combfit_8cx0_{i}_best_structure_reorder.pdb',f'./test_8cx0/group_{i}/combfit_8cx0_{i}_best_structure_new.pdb')
          #   reference_sequences = list(get_chain_sequences(align_density_pdb_path).values())
          #   reorder_and_renumber_chains(combfit_align_output,combfit_reorder_align_output, reference_sequences)
          # combfit_reorder_align_output = f'./combfit_dock.pdb'
          # combfitter.pdb_path = align_density_pdb_path
          # if heter:
          #   chain_path = os.path.join(align_density_path, f'{model_name}_{it}_{num_recycle_cur_iter}')
          #   splitchain(align_density_pdb_path,name,chain_path)
          #   combfitter.pdb = heteroligomer_list_construct(chain_path)
          #   combfitter.output_path = chain_path
          # else:
          #   combfitter.output_path = align_density_path
          # combfitter.pdb_name_used = f'combfit_{model_name}_{it}_{num_recycle_cur_iter}'
          # combfitter.docking_pipeline()
          # logging.info(f'combfit input %s,%s,%s',align_density_pdb_path,density_work_dir,mrc_path)
          if it<2:
            # combfit_reorder_align_output = os.path.join(density_work_dir, f'combfit_{model_name}_{it}_{num_recycle_cur_iter}_best_structure_reorder.pdb')
            # reorder_align_density_pdb_path = os.path.join(align_density_path, f'{model_name}_{it}_{num_recycle_cur_iter}_reorder.pdb')
            # renumber_chains(align_density_pdb_path,reorder_align_density_pdb_path)
            # cmd = f'bash pipeline_integrate.sh {reorder_align_density_pdb_path} {density_work_dir} {resolution} {mrc_path} combfit_{model_name}_{it}_{num_recycle_cur_iter}'
            # t1 = time.time()
            # _ = subprocess.run(cmd, shell=True)
            # # splitchain(align_density_pdb_path,name,f'combfit_{model_name}_{it}_{num_recycle_cur_iter}')
            # t2 = time.time()
            # logging.info(f'combfit consume %ds',t2-t1)
            # combfit_align_output = os.path.join(density_work_dir, f'combfit_{model_name}_{it}_{num_recycle_cur_iter}_best_structure.pdb')
            # # tmp_align_output = os.path.join(density_work_dir, f'combfit_{model_name}_{it}_{num_recycle_cur_iter}_best_structure_tmp.pdb')
            # combfit_reorder_align_output = os.path.join(density_work_dir, f'combfit_{model_name}_{it}_{num_recycle_cur_iter}_best_structure_reorder.pdb')
            # # os.system(f'python3 alphafold/data/tools/20240710_Combfit_heteoligomer.py -pdb_path {align_density_pdb_path} -mrc_path {mrc_path} -resolution {resolution} -output_path {combfit_align_output}')
            # # renumber_chains(combfit_align_output,combfit_reorder_align_output)
            # reference_sequences = list(get_chain_sequences(align_density_pdb_path).values())
            # reorder_and_renumber_chains(combfit_align_output,combfit_reorder_align_output, reference_sequences)
            # renumber_chains(tmp_align_output,combfit_reorder_align_output)
            # combfit_align_output = os.path.join(density_work_dir, f'combfit_{model_name}_{it}_{num_recycle_cur_iter}_best_structure.pdb')
            # with open(combfit_align_output, 'r') as f:
            #   combfit_pdb = f.readlines()
            #   protein_strings = ''.join(combfit_pdb)
            #   combfit_protein = protein.from_pdb_string(protein_strings)
            #   # replace features with combfit positions
            
            # if transform_em == 'replace':
            #   prev['prev_pos'] = combfit_protein.atom_positions
            # elif transform_em == 'interpolate':
            #   interpolate_output = os.path.join(align_density_path, f'interpolate_{model_name}_{it+1}_{num_recycle_cur_iter}.pdb')
            #   interpolate_structures(combfit_reorder_align_output,align_density_pdb_path,0.2,interpolate_output)
            #   with open(interpolate_output, 'r') as f:
            #     protein_strings = ''.join(f.readlines())
            #     interpolate_protein = protein.from_pdb_string(protein_strings)
            #   prev['prev_pos'] = interpolate_protein.atom_positions
            # elif transform_em == 'restraints':
            feat['sbr'],feat['sbr_mask'],feat['interface_mask'] = copy.deepcopy(restraints0['sbr']),copy.deepcopy(restraints0['sbr_mask']),copy.deepcopy(restraints0['interface_mask'])
              # restraints = {i:feat[i] for i in ['sbr','sbr_mask','interface_mask','asym_id']}
            feat = extract_restraints(align_density_pdb_path,combfit_clean_output,30,feat)
          else:
            is_continue = False
            # t_0 = time.time()
          # restraints = {i:feat[i] for i in ['sbr','sbr_mask','interface_mask','asym_id']}
          # rm_num, break_num, max_nb_dist, recall, recall_conf, viol_num, max_viol_dist = filter_restraints(restraints, restraints0, unrelaxed_protein, update=False,nbdist_ca_thre=nbdist_ca_thre, max_rm_ratio=max_rm_ratio, viol_thre=viol_thre, mask_terminal_residues=mask_terminal_residues)
          # rest = int(restraints['interface_mask'].sum() + restraints['sbr_mask'].sum()/2)
          # ranking_score = result['ranking_confidence']
          # confidence = result["mean_plddt"]
          # mydict = {
          #     'Iter': it,
          #     'Conf': confidence,
          #     'RankScore': ranking_score,
          #     'Total': rm_num+rest,
          #     'Remove': rm_num,
          #     'Rest': rest,
          #     'MaxNbDist': max_nb_dist,
          #     'BreakNum': break_num,
          #     'Recall': recall,
          #     # 'RecallByConf': recall_conf,
          #     'Recycle_num': num_recycle_cur_iter,
          #     'Diff': result['tol'],
          #     'ViolNum': int(viol_num),
          #     'MaxViolDist': round(max_viol_dist, 2),
          #     # 'Time': round(time.time()-t_0, 2)
          #   }
          # mydicts.append(mydict)
          num_recycle_cur_iter = 0
          del unrelaxed_protein
      return result,feat,it,num_recycle_cur_iter,prev, is_continue,restraints
        
    prediction_result,recycle = model_runner.predict(processed_feature_dict,restraints,random_seed=model_random_seed,callback=callback)
    logging.info('Recycle %d finished', recycle)
    df = pd.DataFrame(mydicts)
    df.to_csv(os.path.join(output_dir, f'{model_name}_info.csv'),sep='\t',index=False)
    plddt = prediction_result['plddt']
    # _save_confidence_json_file(plddt, output_dir, model_name)
    # ranking_confidence= prediction_result['ranking_confidence']
    plddt_b_factors = np.repeat(
      plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode)
  # Rank by model confidence.
    # ranking_confidences[model_name] = prediction_result['mean_plddt']
    ranking_confidences[model_name] = prediction_result['ranking_confidence']
    unrelaxed_proteins[model_name] = unrelaxed_protein
    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}_final.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(unrelaxed_pdbs[model_name])
    
  ranked_order = [
      model_name for model_name, confidence in
      sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)]


  for idx, model_name in enumerate(ranked_order):
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
      f.write(unrelaxed_pdbs[model_name])

  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    ranking_confidences = {model_name:confidence.tolist() for model_name, confidence in ranking_confidences.items()}
    # logging.info('ranking_confidences %s',type(ranking_confidences) )
    f.write(json.dumps(
        {label: ranking_confidences, 'order': ranked_order}, indent=4))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for tool_name in (
      'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
    if not FLAGS[f'{tool_name}_binary_path'].value:
      raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                       'sure it is installed on your system.')

  use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
  _check_flag('small_bfd_database_path', 'db_preset',
              should_be_set=use_small_bfd)
  _check_flag('bfd_database_path', 'db_preset',
              should_be_set=not use_small_bfd)
  _check_flag('uniref30_database_path', 'db_preset',
              should_be_set=not use_small_bfd)

  run_multimer_system = 'multimer' in FLAGS.model_preset
  model_type = 'Multimer' if run_multimer_system else 'Monomer'
  _check_flag('pdb70_database_path', 'model_preset',
              should_be_set=not run_multimer_system)
  _check_flag('pdb_seqres_database_path', 'model_preset',
              should_be_set=run_multimer_system)
  _check_flag('uniprot_database_path', 'model_preset',
              should_be_set=run_multimer_system)

  if FLAGS.model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  if run_multimer_system:
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        database_path=FLAGS.pdb_seqres_database_path)
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
  else:
    template_searcher = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        databases=[FLAGS.pdb70_database_path])
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  monomer_data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniref30_database_path=FLAGS.uniref30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd,
      use_precomputed_msas=FLAGS.use_precomputed_msas)

  if run_multimer_system:
    num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        uniprot_database_path=FLAGS.uniprot_database_path,
        use_precomputed_msas=FLAGS.use_precomputed_msas)
  else:
    num_predictions_per_model = 1
    data_pipeline = monomer_data_pipeline

  model_runners = {}
  model_names = [config.MODEL_PRESETS[FLAGS.model_preset]]
  model_names = model_names[0]
  for model_name in model_names:
    model_config = config.model_config(model_name)
    model_config.model.max_recycles = 40
    if run_multimer_system:
      model_config.model.num_ensemble_eval = num_ensemble
    else:
      model_config.data.eval.num_ensemble = num_ensemble
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params)
    for i in range(num_predictions_per_model):
      model_runners[f'{model_name}_pred_{i}'] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  amber_relaxer = relax.AmberRelaxation(
      max_iterations=RELAX_MAX_ITERATIONS,
      tolerance=RELAX_ENERGY_TOLERANCE,
      stiffness=RELAX_STIFFNESS,
      exclude_residues=RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
      use_gpu=FLAGS.use_gpu_relax)

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_runners))
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure for each of the sequences.
  for i, fasta_path in enumerate(FLAGS.fasta_paths):
    fasta_name = fasta_names[i]
    predict_structure(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        feature_path=FLAGS.feature_pickle,
        restraints_path=FLAGS.restraints_pickle,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed,
        models_to_relax=FLAGS.models_to_relax,
        model_type=model_type,
        mrc_path = FLAGS.mrc_path,
        resolution = FLAGS.resolution,
        name = FLAGS.name
    )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'data_dir',
      'uniref90_database_path',
      'mgnify_database_path',
      'template_mmcif_dir',
      'max_template_date',
      'obsolete_pdbs_path',
      'use_gpu_relax',
  ])

  app.run(main)
