#!/bin/bash

setup_params () {
  tool=$1
  param_file=$runs/${tool}_params.json
  cp massivefold/parallelization/${tool}_params.json $param_file
  if [ $tool == "AFmassive" ]; then
    db=$alphafold_databases
  elif [ $tool == "AlphaFold3" ]; then
    db=$alphafold3_databases
  elif [ $tool == "ColabFold" ]; then
    db=$colabfold_databases
  fi

  # parameters auto setting
  params_with_paths=$(cat $param_file | python3 -c "
import json
import sys

params = json.load(sys.stdin)

if '$tool' == 'AFmassive':
  params['massivefold']['run_massivefold'] = 'run_AFmassive.py'
if '$tool' == 'AlphaFold3':
  params['massivefold']['run_massivefold'] = 'run_alphafold.py'
params['massivefold']['run_massivefold_plots'] = 'massivefold_plots.py'
params['massivefold']['data_dir'] = '$(realpath $db)'
params['massivefold']['jobfile_templates_dir'] = '../massivefold/parallelization/templates'
params['massivefold']['scripts_dir'] = '../massivefold/parallelization'
params['massivefold']['jobfile_headers_dir'] = './headers'
params['massivefold']['output_dir'] = './output'
params['massivefold']['logs_dir'] = './log'
params['massivefold']['input_dir'] = './input'

key_order = ['run_massivefold', 'run_massivefold_plots', 'data_dir', 'uniref_database', \
'jobfile_headers_dir', 'jobfile_templates_dir', 'scripts_dir', 'output_dir', \
'logs_dir', 'input_dir', 'models_to_use', 'pkl_format']
sorted_keys = sorted(params['massivefold'], key=lambda x: key_order.index(x))
mf_params_ordered = {key: params['massivefold'][key] for key in sorted_keys}

params['massivefold'] = mf_params_ordered
with open('$param_file', 'w') as params_output:
    json.dump(params, params_output, indent=4)")

  cat $param_file
    tool=$1
    echo "$tool"
}


# set file tree
db_af="true"
db_af3="true"
db_cf="true"
runs=massivefold_runs
mkdir -p $runs/input
mkdir $runs/output
mkdir $runs/log
cp examples/H1140.fasta $runs/input

# scripts and files for each pipeline (currently only AFmassive)
cp massivefold/run_massivefold.sh $runs
cp -r massivefold/parallelization/headers $runs

if [[ $host_is_jeanzay == "true" ]]; then
  mkdir $HOME/af3_datadir/
  ln -s $ALPHAFOLD3DB/* $HOME/af3_datadir/
  cp massivefold/parallelization/jeanzay_AFmassive_params.json $runs/AFmassive_params.json
  cp massivefold/parallelization/jeanzay_AlphaFold3_params.json $runs/AlphaFold3_params.json
  cp massivefold/parallelization/jeanzay_ColabFold_params.json $runs/ColabFold_params.json
  echo "Taking Jean Zay's prebuilt headers and renaming them."
  mv $runs/headers/example_header_alignment_jeanzay.slurm $runs/headers/alignment.slurm
  mv $runs/headers/example_header_jobarray_jeanzay.slurm $runs/headers/jobarray.slurm
  mv $runs/headers/example_header_post_treatment_jeanzay.slurm $runs/headers/post_treatment.slurm
  exit 1
fi

if [[ $db_af == "true" ]]; then
  setup_params "AFmassive"
fi
if [[ $db_af3 == "true" ]]; then
  setup_params "AlphaFold3"
fi
if [[ $db_cf == "true" ]]; then
  setup_params "ColabFold"
fi
