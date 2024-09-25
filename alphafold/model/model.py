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

"""Code for constructing the model."""
from typing import Any, Mapping, Optional, Union

from absl import logging
from alphafold.common import confidence
from alphafold.model import features
from alphafold.model import modules
from alphafold.model import modules_multimer
import haiku as hk
import jax
import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf
import tree


def get_confidence_metrics(
    prediction_result: Mapping[str, Any],
    multimer_mode: bool) -> Mapping[str, Any]:
  """Post processes prediction_result to get confidence metrics."""
  confidence_metrics = {}
  confidence_metrics['plddt'] = confidence.compute_plddt(
      prediction_result['predicted_lddt']['logits'])
  if 'predicted_aligned_error' in prediction_result:
    confidence_metrics.update(confidence.compute_predicted_aligned_error(
        logits=prediction_result['predicted_aligned_error']['logits'],
        breaks=prediction_result['predicted_aligned_error']['breaks']))
    confidence_metrics['ptm'] = confidence.predicted_tm_score(
        logits=prediction_result['predicted_aligned_error']['logits'],
        breaks=prediction_result['predicted_aligned_error']['breaks'],
        asym_id=None)
    if multimer_mode:
      # Compute the ipTM only for the multimer model.
      confidence_metrics['iptm'] = confidence.predicted_tm_score(
          logits=prediction_result['predicted_aligned_error']['logits'],
          breaks=prediction_result['predicted_aligned_error']['breaks'],
          asym_id=prediction_result['predicted_aligned_error']['asym_id'],
          interface=True)
      confidence_metrics['ranking_confidence'] = (
          0.8 * confidence_metrics['iptm'] + 0.2 * confidence_metrics['ptm'])

  if not multimer_mode:
    # Monomer models use mean pLDDT for model ranking.
    confidence_metrics['ranking_confidence'] = np.mean(
        confidence_metrics['plddt'])

  return confidence_metrics


class RunModel:
  """Container for JAX model."""

  def __init__(self,
               config: ml_collections.ConfigDict,
               params: Optional[Mapping[str, Mapping[str, jax.Array]]] = None):
    self.config = config
    self.params = params
    self.multimer_mode = config.model.global_config.multimer_mode

    if self.multimer_mode:
      def _forward_fn(batch):
        model = modules_multimer.AlphaFold(self.config.model)
        return model(
            batch,
            is_training=False)
    else:
      def _forward_fn(batch):
        model = modules.AlphaFold(self.config.model)
        return model(
            batch,
            is_training=False,
            compute_loss=False,
            ensemble_representations=True)

    self.apply = jax.jit(hk.transform(_forward_fn).apply)
    self.init = jax.jit(hk.transform(_forward_fn).init)
  def init_params(self, feat: features.FeatureDict, random_seed: int = 0):
    """Initializes the model parameters.

    If none were provided when this class was instantiated then the parameters
    are randomly initialized.

    Args:
      feat: A dictionary of NumPy feature arrays as output by
        RunModel.process_features.
      random_seed: A random seed to use to initialize the parameters if none
        were set when this class was initialized.
    """
    if not self.params:
      # Init params randomly.
      rng = jax.random.PRNGKey(random_seed)
      self.params = hk.data_structures.to_mutable_dict(
          self.init(rng, feat))
      logging.warning('Initialized parameters randomly')

  def process_features(
      self,
      raw_features: Union[tf.train.Example, features.FeatureDict],
      random_seed: int) -> features.FeatureDict:
    """Processes features to prepare for feeding them into the model.

    Args:
      raw_features: The output of the data pipeline either as a dict of NumPy
        arrays or as a tf.train.Example.
      random_seed: The random seed to use when processing the features.

    Returns:
      A dict of NumPy feature arrays suitable for feeding into the model.
    """

    if self.multimer_mode:
      return raw_features

    # Single-chain mode.
    if isinstance(raw_features, dict):
      return features.np_example_to_features(
          np_example=raw_features,
          config=self.config,
          random_seed=random_seed)
    else:
      return features.tf_example_to_features(
          tf_example=raw_features,
          config=self.config,
          random_seed=random_seed)

  def eval_shape(self, feat: features.FeatureDict) -> jax.ShapeDtypeStruct:
    self.init_params(feat)
    logging.info('Running eval_shape with shape(feat) = %s',
                 tree.map_structure(lambda x: x.shape, feat))
    shape = jax.eval_shape(self.apply, self.params, jax.random.PRNGKey(0), feat)
    logging.info('Output shape was %s', shape)
    return shape

  def predict(self,
              feat: features.FeatureDict,
              restraints: features.FeatureDict,
              random_seed: int = 0,
              return_representations: bool = False,
              callback: Any = None) -> Mapping[str, Any]:
              # cryoem_pdb_path: str = None) -> Mapping[str, Any]:
    """Makes a prediction by inferencing the model on the provided features.

    Args:
      feat: A dictionary of NumPy feature arrays as output by
        RunModel.process_features.
      random_seed: The random seed to use when running the model. In the
        multimer model this controls the MSA sampling.

    Returns:
      A dictionary of model outputs.
    """
     # get shapes
    self.init_params(feat)
    aatype = feat["aatype"]
    num_iters = self.config.model.num_recycle + 1
    if self.multimer_mode:
      L = aatype.shape[0]
    else:
      num_ensemble = self.config.data.eval.num_ensemble
      L = aatype.shape[1]

    # initialize
    zeros = lambda shape: np.zeros(shape, dtype=np.float16)

    prev = {'prev_msa_first_row': zeros([L,256]),
            'prev_pair':          zeros([L,L,128]),
            'prev_pos':           zeros([L,37,3])}

    # if 'prev_pair'  in feat:
    #   prev['prev_pair'] = feat['prev_pair']
    # if 'prev_pos'  in feat:
    #   prev['prev_pos'] = feat['prev_pos']

    def run(key, feat, prev):
      def _jnp_to_np(x):
        for k, v in x.items():
          if isinstance(v, dict):
            x[k] = _jnp_to_np(v)
          else:
            x[k] = np.asarray(v,np.float16)
        return x
      result = _jnp_to_np(self.apply(self.params, key, {**feat, "prev":prev}))
      prev = result.pop("prev")
      return result, prev
    
    key = jax.random.PRNGKey(random_seed)
    it = 0 
    local_it = 0
    is_continue = True
    # iterate through recyckes
    for r in range(num_iters):
      if r > self.config.model.num_recycle:
        break
      if not is_continue:
        break
        # grab subset of features
      if self.multimer_mode:
        sub_feat = feat
      else:
        s = r * num_ensemble
        e = (r+1) * num_ensemble
        sub_feat = jax.tree_map(lambda x:x[s:e], feat)

      # run
      key, sub_key = jax.random.split(key)
      result, prev = run(sub_key, sub_feat, prev)

      if return_representations:

        result["representations"] = {"pair":   prev["prev_pair"],
                                      "single": prev["prev_msa_first_row"]}
        
      if callback is not None: 
        result,feat,it, local_it, prev ,is_continue,restraints = callback(result, restraints, r, feat, it, local_it, prev)
      #  prev['prev_pos'] = result["structure_module"]['final_atom_positions']

        # decide when to stop
      # if result["ranking_confidence"] > self.config.model.stop_at_score:
      #   break
      # if r >= self.config.model.num_recycle:
      #   break
      # if not is_continue:
      #   break

    logging.info('Output shape was %s', tree.map_structure(lambda x: x.shape, result))
    # logging.info('Early stop at %d iter', r+1)
    return result, r
    
    
    
    
    
    # self.init_params(feat)
    # logging.info('Running predict with shape(feat) = %s',
    #              tree.map_structure(lambda x: x.shape, feat))
    # result = self.apply(self.params, jax.random.PRNGKey(random_seed), feat)

    # # This block is to ensure benchmark timings are accurate. Some blocking is
    # # already happening when computing get_confidence_metrics, and this ensures
    # # all outputs are blocked on.
    # jax.tree_map(lambda x: x.block_until_ready(), result)
    # result.update(
    #     get_confidence_metrics(result, multimer_mode=self.multimer_mode))
    # logging.info('Output shape was %s',
    #              tree.map_structure(lambda x: x.shape, result))
    # return result
