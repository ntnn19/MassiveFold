#!/usr/bin/env python

import math
import json
import numpy as np
from copy import deepcopy
from absl import app, flags

# Define the flags
flags.DEFINE_integer('predictions_per_model', 25, 
                     'Choose the number of predictions inferred by each neural network model.')
flags.DEFINE_integer('batch_size', 25, 
                     'Standard size of a prediction batch, if the number of prediction per model\
                       is not a multiple of it, the last batch will be smaller .')
flags.DEFINE_list('models_to_use', None, 'Select the models used for prediction among the five models of each AlphaFold2 version (15 in total).')
flags.DEFINE_string('sequence_name', '', 'Name of the sequence to predict.')
flags.DEFINE_string('run_name', '', 'Name of the run.')
flags.DEFINE_string('path_to_parameters', '', 'Parameters to use, contains models_to_use')
flags.DEFINE_enum('tool', 'AFmassive', ['AFmassive', 'AlphaFold3', 'ColabFold'], 'Specify the tool that you want to use')

# Add flags for parallel processing
flags.DEFINE_bool('parallel', False, 'Flag to enable parallel job distribution.')
flags.DEFINE_string('gpu_nodes_file', '', 'Tab-separated file containing GPU nodes info.')
flags.DEFINE_string('jobs_file', '', 'File containing the list of jobs.')

FLAGS = flags.FLAGS

def batches_per_model(pred_nb_per_model:int):
  """Split the predictions into batches based on the batch size."""
  opt_batch_nb = math.ceil(pred_nb_per_model/FLAGS.batch_size)
  batch_sizes = []
  for _ in range(1, opt_batch_nb+1):
    # split total by batch of the same size, the remaining is a single smaller batch
    if pred_nb_per_model - FLAGS.batch_size >= 0:
      pred_nb_per_model -= FLAGS.batch_size
      batch_sizes.append(FLAGS.batch_size)
    else:
      batch_sizes.append(pred_nb_per_model)
  batch_edges = list(np.cumsum([0] + batch_sizes))
  one_model_batches = {i+1: {'start': str(batch_edges[i]), 'end': str(batch_edges[i+1]-1)} for i in range(len(batch_edges)-1)}
  return one_model_batches

def batches_all_models(batches_unit, all_models):
  """Distribute the batches across all models."""
  batches = {}

  for i, model in enumerate(all_models):
    unadded_batch = deepcopy(batches_unit)
    for batch in unadded_batch:
      unadded_batch[batch].update({'model': model})
      batches[str(batch+i*len(batches_unit)-1)] = unadded_batch[batch]
      
  return batches

def distribute_jobs(gpu_nodes_file, jobs_file, output_prefix):
    """Distribute jobs across GPU nodes based on available GPUs and create job files."""
    
    # Read GPU node information
    gpu_nodes = {}
    with open(gpu_nodes_file, 'r') as f:
        next(f)
        for line in f:
            cols = line.strip().split("\t")
            node_name = cols[0]
            gpus = int(cols[3])  # Number of GPUs on the node
            gpu_nodes[node_name] = gpus

    # Read the job list
    print("jobs_file=",jobs_file)
    with open(jobs_file, 'r') as f:
        jobs = f.read().strip().splitlines()

    # Number of GPUs across all nodes
    total_gpus = sum(gpu_nodes.values())
    
    # Calculate the number of jobs per GPU node
    jobs_per_node = {}
    job_index = 0
    i=0
    for node, gpus in gpu_nodes.items():
        assigned_jobs = jobs[job_index:job_index+gpus]
        job_index += gpus

        # Write each node's job list to a separate file
        job_filename = f"{output_prefix}_{FLAGS.run_name}_jobs_{i}.txt"
        with open(job_filename, 'w') as job_file:
            job_file.write("\n".join(assigned_jobs) + "\n")

        jobs_per_node[node] = job_filename  # Store filename instead of job list
        i+=1

    return jobs_per_node

def main(argv):
    if not FLAGS.path_to_parameters:
        raise ValueError('--path_to_parameters is required')

    if FLAGS.parallel:
        if not FLAGS.gpu_nodes_file or not FLAGS.jobs_file:
            raise ValueError('--gpu_nodes_file and --jobs_file are required when parallel flag is enabled.')

        # Distribute jobs to GPU nodes and create sub-job files
        jobs_per_node = distribute_jobs(FLAGS.gpu_nodes_file, FLAGS.jobs_file, FLAGS.sequence_name)

        # Output the job distribution to a JSON file
        output_json = f"{FLAGS.sequence_name}_{FLAGS.run_name}_jobs_per_node.json"
        with open(output_json, "w") as json_output:
            json.dump(jobs_per_node, json_output, indent=4)

    else:
        # Continue with the model batching logic when parallel flag is not set
        with open(FLAGS.path_to_parameters, 'r') as params:
            all_params = json.load(params)

        if FLAGS.tool == "AFmassive":
            models = all_params['massivefold']['models_to_use']
            tool_code = "AFM"
        elif FLAGS.tool == "AlphaFold3":
            models = ["AlphaFold3"]
            tool_code = "AF3"
        elif FLAGS.tool == "ColabFold":
            models = all_params['massivefold']['models_to_use']
            tool_code = "CF"

        model_preset = all_params[f'{tool_code}_run'][f'model_preset']  

        if FLAGS.tool == "AlphaFold3":
            model_names = ["AlphaFold3"]
        elif model_preset == 'multimer':
            model_names = [
            'model_1_multimer_v1',
            'model_2_multimer_v1',
            'model_3_multimer_v1',
            'model_4_multimer_v1',
            'model_5_multimer_v1',
            'model_1_multimer_v2',
            'model_2_multimer_v2',
            'model_3_multimer_v2',
            'model_4_multimer_v2',
            'model_5_multimer_v2',
            'model_1_multimer_v3',
            'model_2_multimer_v3',
            'model_3_multimer_v3',
            'model_4_multimer_v3',
            'model_5_multimer_v3'
            ]
        elif model_preset == 'monomer_ptm':
            model_names = [
            'model_1_ptm',
            'model_2_ptm',
            'model_3_ptm',
            'model_4_ptm',
            'model_5_ptm',
            ]

        if models:
            model_names = [model for model in model_names if model in models]
        if FLAGS.models_to_use:
            model_names = [model for model in model_names if model in FLAGS.models_to_use]

        print(f"Running inference on models: {(', ').join(model_names)}") 
        print(f"Running {FLAGS.predictions_per_model} predictions on each of the {len(model_names)} models")
        print(f"Total prediction number: {FLAGS.predictions_per_model * len(model_names)}")

        # Divide the predictions in batches 
        per_model_batches = batches_per_model(pred_nb_per_model=FLAGS.predictions_per_model)
        # Distribute the batches on all models
        all_model_batches = batches_all_models(per_model_batches, model_names)

        # Output the batch assignments to a JSON file
        with open(f"{FLAGS.sequence_name}_{FLAGS.run_name}_batches.json", "w") as json_output:
            json.dump(all_model_batches, json_output, indent=4)

if __name__ == "__main__":
    app.run(main)
