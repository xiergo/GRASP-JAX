import os
# import re
import sys
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from Bio import BiopythonDeprecationWarning
warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)
from pathlib import Path
# from colabfold.download import download_alphafold_params, default_data_dir
from colabfold.utils import setup_logging
from colabfold.batch import get_queries
from colabfold.batch import my_run as run
# from colabfold.plot import plot_msa_v2
# from colabfold.colabfold import plot_protein
# import matplotlib.pyplot as plt

def search_features_colab(fasta_path, outdir):
  with open(fasta_path, "r"):
    seqs = [line.strip() for line in open(fasta_path, "r")]
    seqs = [s for s in seqs[1::2]]
  query_sequence = ':'.join(seqs)
  jobname = os.path.basename(fasta_path).split(".fas")[0]

  # query_sequence = 'PIAQIHILEGRSDEQKETLIREVSEAISRS:LDAPLTSVRVIITEMAKGHFGIGGELASK' #@param {type:"string"}
  # outdir = '../test222/colab_features_search' #@param {type:"string"}
  # jobname = 'test'
  # number of models to use
  num_relax = 0 #@param [0, 1, 5] {type:"raw"}
  #@markdown - specify how many of the top ranked structures to relax using amber
  template_mode = "pdb100" #@param ["none", "pdb100","custom"]
  #@markdown - `none` = no template information is used. `pdb100` = detect templates in pdb100 (see [notes](#pdb100)). `custom` - upload and search own templates (PDB or mmCIF format, see [notes](#custom_templates))

  use_amber = num_relax > 0

  # remove whitespaces
  query_sequence = "".join(query_sequence.split())


  # make directory to save results
  os.makedirs(outdir, exist_ok=True)

  # save queries
  queries_path = os.path.join(outdir, f"{jobname}.csv")
  with open(queries_path, "w") as text_file:
    text_file.write(f"id,sequence\n{jobname},{query_sequence}")

  if template_mode == "pdb100":
    use_templates = True
    custom_template_path = None
  elif template_mode == "custom":
    raise NotImplementedError("Custom templates not implemented yet")
    # custom_template_path = os.path.join(jobname,f"template")
    # os.makedirs(custom_template_path, exist_ok=True)
    # uploaded = files.upload()
    # use_templates = True
    # for fn in uploaded.keys():
    #   os.rename(fn,os.path.join(custom_template_path,fn))
  else:
    custom_template_path = None
    use_templates = False

  print("outdir",outdir)
  print("sequence",query_sequence)
  print("length",len(query_sequence.replace(":","")))


  #@markdown ### MSA options (custom MSA upload, single sequence, pairing mode)
  msa_mode = "mmseqs2_uniref_env" #@param ["mmseqs2_uniref_env", "mmseqs2_uniref","single_sequence","custom"]
  pair_mode = "unpaired_paired" #@param ["unpaired_paired","paired","unpaired"] {type:"string"}
  #@markdown - "unpaired_paired" = pair sequences from same species + unpaired MSA, "unpaired" = seperate MSA for each chain, "paired" - only use paired sequences.

  # decide which a3m to use
  if "mmseqs2" in msa_mode:
    a3m_file = os.path.join(outdir,f"{jobname}.a3m")

  elif msa_mode == "custom":
    raise NotImplementedError("Custom MSA not implemented yet")
    # a3m_file = os.path.join(jobname,f"{jobname}.custom.a3m")
    # if not os.path.isfile(a3m_file):
    #   custom_msa_dict = files.upload()
    #   custom_msa = list(custom_msa_dict.keys())[0]
    #   header = 0
    #   import fileinput
    #   for line in fileinput.FileInput(custom_msa,inplace=1):
    #     if line.startswith(">"):
    #        header = header + 1
    #     if not line.rstrip():
    #       continue
    #     if line.startswith(">") == False and header == 1:
    #        query_sequence = line.rstrip()
    #     print(line, end='')

    #   os.rename(custom_msa, a3m_file)
    #   queries_path=a3m_file
    #   print(f"moving {custom_msa} to {a3m_file}")

  else:
    a3m_file = os.path.join(outdir,f"{jobname}.single_sequence.a3m")
    with open(a3m_file, "w") as text_file:
      text_file.write(">1\n%s" % query_sequence)

  #@markdown ### Advanced settings
  model_type = "alphafold2_multimer_v1" #@param ["auto", "alphafold2_ptm", "alphafold2_multimer_v1", "alphafold2_multimer_v2", "alphafold2_multimer_v3", "deepfold_v1", "alphafold2"]
  #@markdown - if `auto` selected, will use `alphafold2_ptm` for monomer prediction and `alphafold2_multimer_v3` for complex prediction.
  #@markdown Any of the mode_types can be used (regardless if input is monomer or complex).
  num_recycles = "3" #@param ["auto", "0", "1", "3", "6", "12", "24", "48"]
  #@markdown - if `auto` selected, will use `num_recycles=20` if `model_type=alphafold2_multimer_v3`, else `num_recycles=3` .
  recycle_early_stop_tolerance = "auto" #@param ["auto", "0.0", "0.5", "1.0"]
  #@markdown - if `auto` selected, will use `tol=0.5` if `model_type=alphafold2_multimer_v3` else `tol=0.0`.
  relax_max_iterations = 200 #@param [0, 200, 2000] {type:"raw"}
  #@markdown - max amber relax iterations, `0` = unlimited (AlphaFold2 default, can take very long)
  pairing_strategy = "greedy" #@param ["greedy", "complete"] {type:"string"}
  #@markdown - `greedy` = pair any taxonomically matching subsets, `complete` = all sequences have to match in one line.
  calc_extra_ptm = False #@param {type:"boolean"}
  #@markdown - return pairwise chain iptm/actifptm

  #@markdown #### Sample settings
  #@markdown -  enable dropouts and increase number of seeds to sample predictions from uncertainty of the model.
  #@markdown -  decrease `max_msa` to increase uncertainity
  max_msa = "auto" #@param ["auto", "512:1024", "256:512", "64:128", "32:64", "16:32"]
  num_seeds = 1 #@param [1,2,4,8,16] {type:"raw"}
  use_dropout = False #@param {type:"boolean"}

  num_recycles = None if num_recycles == "auto" else int(num_recycles)
  recycle_early_stop_tolerance = None if recycle_early_stop_tolerance == "auto" else float(recycle_early_stop_tolerance)
  if max_msa == "auto": max_msa = None

  #@markdown #### Save settings
  save_all = False #@param {type:"boolean"}
  save_recycles = False #@param {type:"boolean"}
  save_to_google_drive = False #@param {type:"boolean"}
  #@markdown -  if the save_to_google_drive option was selected, the result zip will be uploaded to your Google Drive
  dpi = 200 #@param {type:"integer"}
  #@markdown - set dpi for image resolution

  # if save_to_google_drive:
  #   from pydrive2.drive import GoogleDrive
  #   from pydrive2.auth import GoogleAuth
  #   from google.colab import auth
  #   from oauth2client.client import GoogleCredentials
  #   auth.authenticate_user()
  #   gauth = GoogleAuth()
  #   gauth.credentials = GoogleCredentials.get_application_default()
  #   drive = GoogleDrive(gauth)
  #   print("You are logged into Google Drive and are good to go!")

  # @markdown Don't forget to hit `Runtime` -> `Run all` after updating the form.


  #@title Run Prediction
  display_images = True #@param {type:"boolean"}

  

  use_amber=False
  # def input_features_callback(input_features):
  #   if display_images:
  #     plot_msa_v2(input_features)
  #     plt.show()
  #     plt.close()

  # def prediction_callback(protein_obj, length,
  #                         prediction_result, input_features, mode):
  #   model_name, relaxed = mode
  #   if not relaxed:
  #     if display_images:
  #       fig = plot_protein(protein_obj, Ls=length, dpi=150)
  #       plt.show()
  #       plt.close()

  result_dir = outdir
  log_filename = os.path.join(jobname,"log.txt")
  setup_logging(Path(log_filename))

  queries, is_complex = get_queries(queries_path)
  # model_type = set_model_type(is_complex, model_type)

  if "multimer" in model_type and max_msa is not None:
    use_cluster_profile = False
  else:
    use_cluster_profile = True

  # download_alphafold_params(model_type, Path("."))
  results = run(
      queries=queries,
      result_dir=result_dir,
      use_templates=use_templates,
      custom_template_path=custom_template_path,
      num_relax=num_relax,
      msa_mode=msa_mode,
      model_type=model_type,
      num_models=0,
      num_recycles=num_recycles,
      relax_max_iterations=relax_max_iterations,
      recycle_early_stop_tolerance=recycle_early_stop_tolerance,
      num_seeds=num_seeds,
      use_dropout=use_dropout,
      model_order=[1,2,3,4,5],
      is_complex=is_complex,
      data_dir=Path("."),
      keep_existing_results=False,
      rank_by="auto",
      pair_mode=pair_mode,
      pairing_strategy=pairing_strategy,
      stop_at_score=float(100),
      prediction_callback=None,
      dpi=dpi,
      zip_results=False,
      save_all=save_all,
      max_msa=max_msa,
      use_cluster_profile=use_cluster_profile,
      input_features_callback=None,
      save_recycles=save_recycles,
      user_agent="colabfold/google-colab-main",
      calc_extra_ptm=calc_extra_ptm,
  )
  return f'{outdir}/features.pkl'


if __name__ == '__main__':

  fasta_path = sys.argv[1]
  outdir = sys.argv[2]
  features_path = search_features_colab(fasta_path, outdir)
  print(features_path)
