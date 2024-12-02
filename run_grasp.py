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
from alphafold.data.tools.align import interpolate_structures,extract_restraints
import jax.numpy as jnp
import numpy as np
import tree
# Internal import (7716).
# from alphafold.data.tools.powerfit_api import *
# logging.set_verbosity(logging.INFO)


@enum.unique
class ModelsToRelax(enum.Enum):
  ALL = 0
  BEST = 1
  NONE = 2

flags.DEFINE_string(
    'fasta_path', None, 'Path to FASTA file')
flags.DEFINE_string('mrc_path', None, 'Path to the mrc file of density map')
flags.DEFINE_float('resolution', 10, 'Resolution of the mrc file')
# flags.DEFINE_string('name', None, 'Name of the protein')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_integer('iter_num', 5,'Maximum iteration for iterative restraint filtering.')
flags.DEFINE_string('mode', 'normal', 'The mode of running GRASP, "normal" or "quick".')
flags.DEFINE_string('feature_pickle', None, 'Path to the feature dictionary generated using AlphaFold-multimer\'s protocal. If not specified, other arguments used for generating features will be required.')
flags.DEFINE_string('restraints_pickle', None, 'Path to a restraint pickle file. If not provided, inference will be done without restraints.')
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
# flags.DEFINE_enum('model_preset', 'multimer',
#                   ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer','multimer-1','multimer-2','multimer-3','multimer-4','multimer-5'],
#                   'Choose preset model configuration - the monomer model, '
#                   'the monomer model with extra ensembling, monomer model with '
#                   'pTM head, or multimer model')
# flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
#                      'to obtain a timing that excludes the compilation time, '
#                      'which should be more indicative of the time required for '
#                      'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('num_multimer_predictions_per_model', 5, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. ')
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
MODEL_PRESET='multimer'
BENCHMARK=False


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
    fasta_path=None,
    fasta_name=None,
    output_dir_base=None,
    feature_path=None,
    restraints_path=None,
    mode='normal',
    iter_num=5,
    data_pipeline=None,
    model_runners=None,
    amber_relaxer=None,
    benchmark=False,
    random_seed=0,
    models_to_relax=None,
    model_type=None,
    mrc_path=None,
    resolution=None,
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
  if not os.path.exists(feature_path):
    logging.info('No feature pickle found, generating features')
    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir)
    # Write out features as a pickled dictionary.
    features_output_path = os.path.join(output_dir, 'features.pkl')
    with open(features_output_path, 'wb') as f:
      pickle.dump(feature_dict, f, protocol=4)
  else:
    logging.info('Reading raw features from %s', feature_path)
    feature_dict = pickle.load(open(feature_path,'rb'))
  
  timings['features'] = time.time() - t_0

  
  BINS = np.arange(4, 33, 1)
  dtype = np.float32
  seq_length = len(feature_dict['aatype'])
  restraints = {'sbr': np.zeros((seq_length, seq_length, len(BINS)+1)).astype(dtype),
    'sbr_mask' : np.zeros((seq_length, seq_length)).astype(dtype),
    'interface_mask': np.zeros(seq_length).astype(dtype)}
  if os.path.exists(restraints_path):
    restraints_input = pickle.load(open(restraints_path,'rb'))
    logging.info(f'read restraints from {restraints_path} successfully')
    if 'asym_id' in restraints_input:
      assert (restraints_input['asym_id'] == feature_dict['asym_id']).all()
    restraints_input = {k: v for k, v in restraints_input.items() if k in ['sbr','sbr_mask','interface_mask']}
    restraints.update(restraints_input)
    rpr_num = int(restraints['sbr_mask'].sum()/2)
    ir_num = int(restraints['interface_mask'].sum())
    logging.info(f'read restraints from {restraints_path} successfully, including {rpr_num} RPR restraints, {ir_num} IR restraints.')

  feature_dict.update(restraints)
  unrelaxed_pdbs = {}
  unrelaxed_proteins = {}
  ranking_confidences = {}
  nbdist_ca_thre=5.0
  viol_thre=5.0
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
  
  if mode == 'quick':
    logging.info('Using quick inference')
    max_recycle_per_iter = 4
  else:
    logging.info('Using normal inference')
    max_recycle_per_iter = 20
  logging.info('At least %d restraints will be used in the final iteration', left_thre)
  
  num_models = len(model_runners)
  
  # Run the models.
  # align_density_path = None
  # logging.info('mrc_path: %s', mrc_path)
  # if mrc_path != "None":
  #   align_density_path = os.path.join(output_dir,'align_density')
  #   if not os.path.exists(align_density_path):
  #     os.mkdir(align_density_path)
  #   combfitter = Combfit(mrc_path = mrc_path, resolution = resolution, pdb_model_path = None, output_path = None, gpu_flag=True, nproc_num=1, name=name)
  
  # transform_em = 'restraints'
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()):
    logging.info('Running model %s', model_name)
    feature_dict.update(restraints0)
    # t_0 = time.time()
    restraints = restraints0.copy()
    mydicts = []
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed)
    # timings[f'process_features_{model_name}'] = time.time() - t_0
    t_0 = time.time()
    # feat0 = processed_feature_dict.copy()
    def callback(result, restraints,recycles,feat,it, num_recycle_cur_iter, prev):
      # if recycles == 0: 
      #   result.pop("tol",None)
      print_line = ""
      for x,y in [["mean_plddt","pLDDT"],["ptm","pTM"],["iptm","ipTM"],["tol","tol"]]:
        if x in result:
          print_line += f" {y}={result[x]:.3g}"
      logging.info("%s recycle=%s, cur recycle=%s%s",model_name,recycles+1, num_recycle_cur_iter+1, print_line)
      num_recycle_cur_iter += 1
      is_continue = True
      zeros = lambda shape: np.zeros(shape, dtype=np.float16)
      L = feat["aatype"].shape[0]
      # if recycles > 0 and result["tol"] < 0.5:
      if num_recycle_cur_iter >= max_recycle_per_iter or (num_recycle_cur_iter > 0 and result["tol"] < 0.5):
          final_atom_mask = result["structure_module"]["final_atom_mask"]
          b_factors = result["plddt"][:, None] * final_atom_mask
          confidence = result["mean_plddt"]
          unrelaxed_protein = protein.from_prediction(
              features=feat,
              result=result, b_factors=b_factors,
              remove_leading_feature_dimension=not model_runner.multimer_mode)
          unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}_{it+1}.pdb')
          with open(unrelaxed_pdb_path, 'w') as f:
              f.write(protein.to_pdb(unrelaxed_protein))
          restraints = {i:feat[i] for i in ['sbr','sbr_mask','interface_mask','asym_id']}
          ranking_score = result['ranking_confidence']
          rm_num, break_num, max_nb_dist, recall, recall_conf, viol_num, max_viol_dist = filter_restraints(restraints, restraints0, unrelaxed_protein, update=True,nbdist_ca_thre=nbdist_ca_thre, max_rm_ratio=max_rm_ratio, viol_thre=viol_thre, mask_terminal_residues=mask_terminal_residues)
          rest = int(restraints['interface_mask'].sum() + restraints['sbr_mask'].sum()/2)
          for i in ['sbr','sbr_mask','interface_mask','asym_id']:
              feat[i] = restraints[i]
          # assert rm_num >=0, rm_num
          mydict = {
            'Iter': it+1,
            'Conf': confidence,
            #'RankScore': ranking_score,
            'Total': rm_num+rest,
            'Remove': rm_num,
            'Rest': rest,
            'MaxNbDist': max_nb_dist,
            'BreakNum': break_num,
            'Recall': recall,
            #'RecallByConf': recall_conf,
            'Recycle_num': num_recycle_cur_iter,
            #'Total_recycle': recycles+1,
            #'Diff': result['tol'],
            'ViolNum': int(viol_num),
            'MaxViolDist': round(max_viol_dist, 2),
            # 'Time': round(time.time()-t_0, 2)
          }
          mydicts.append(mydict)
          
          if mode == 'normal':
            prev = {'prev_msa_first_row': zeros([L,256]),
            'prev_pair': zeros([L,L,128]),
            'prev_pos':  zeros([L,37,3])}

          if (it+1>=iter_num) or (rm_num==0):
              is_continue = False
              
          it += 1
          num_recycle_cur_iter = 0
          del unrelaxed_protein
          result['recall'] = recall
      return result,feat,it,num_recycle_cur_iter,prev, is_continue,restraints

    prediction_result,recycle = model_runner.predict(processed_feature_dict,restraints,random_seed=model_random_seed,callback=callback)
    logging.info('Recycle %d finished', recycle)
    df = pd.DataFrame(mydicts)
    df.to_csv(os.path.join(output_dir, f'{model_name}_info.tsv'),sep='\t',index=False)
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
    ranking_confidences[model_name] = prediction_result['mean_plddt'] + (prediction_result['recall']>=0.3)*1000
    # ranking_confidences[model_name] = prediction_result['ranking_confidence']
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
    # label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    label = 'plddt+1000*(recall>=0.3)'
    ranking_confidences = {model_name:confidence.tolist() for model_name, confidence in ranking_confidences.items()}
    # logging.info('ranking_confidences %s',type(ranking_confidences) )
    f.write(json.dumps(
        {label: ranking_confidences, 'order': ranked_order}, indent=4))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run_multimer_system = 'multimer' in MODEL_PRESET
  model_type = 'Multimer' if run_multimer_system else 'Monomer'
  if run_multimer_system:
    num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
  else:
    num_predictions_per_model = 1
  data_pipeline = None
  if MODEL_PRESET == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1
  
  # Check for duplicate FASTA file names.
  # fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  # if len(fasta_names) != len(set(fasta_names)):
  #   raise ValueError('All FASTA paths must have a unique basename.')

  if not os.path.exists(FLAGS.feature_pickle):
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

    _check_flag('pdb70_database_path', 'model_preset',
                should_be_set=not run_multimer_system)
    _check_flag('pdb_seqres_database_path', 'model_preset',
                should_be_set=run_multimer_system)
    _check_flag('uniprot_database_path', 'model_preset',
                should_be_set=run_multimer_system)

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
      # num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
      data_pipeline = pipeline_multimer.DataPipeline(
          monomer_data_pipeline=monomer_data_pipeline,
          jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
          uniprot_database_path=FLAGS.uniprot_database_path,
          use_precomputed_msas=FLAGS.use_precomputed_msas)
    else:
      # num_predictions_per_model = 1
      data_pipeline = monomer_data_pipeline
  
  model_runners = {}
  model_names = config.MODEL_PRESETS[MODEL_PRESET]
  for model_name in model_names:
    model_config = config.model_config(model_name)
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

  # Predict structure.
  predict_structure(
      fasta_path=FLAGS.fasta_path,
      # fasta_name=fasta_name,
      output_dir_base=FLAGS.output_dir,
      feature_path=FLAGS.feature_pickle,
      mode = FLAGS.mode,
      iter_num=FLAGS.iter_num,
      restraints_path=FLAGS.restraints_pickle,
      data_pipeline=data_pipeline,
      model_runners=model_runners,
      amber_relaxer=amber_relaxer,
      benchmark=BENCHMARK,
      random_seed=random_seed,
      models_to_relax=FLAGS.models_to_relax,
      model_type=model_type,
      mrc_path = FLAGS.mrc_path,
      resolution = FLAGS.resolution,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      # 'fasta_paths',
      'output_dir',
      'data_dir',
      # 'uniref90_database_path',
      # 'mgnify_database_path',
      # 'template_mmcif_dir',
      # 'max_template_date',
      # 'obsolete_pdbs_path',
      # 'use_gpu_relax',
  ])

  app.run(main)
