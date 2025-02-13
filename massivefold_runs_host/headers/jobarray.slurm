#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1            # Number of nodes
#SBATCH --ntasks-per-node=1  # Number of tasks per node
#SBATCH --cpus-per-task=10    # Number of OpenMP threads per task
#SBATCH --hint=nomultithread # Disable hyperthreading
#SBATCH --error=${cwd}/${logs_dir}/${sequence_name}/${run_name}/stderr-jobarray_%a-%x-%j.log
#SBATCH --output=${cwd}/${logs_dir}/${sequence_name}/${run_name}/stdout-jobarray_%a-%x-%j.log
#SBATCH --array=0-$substitute_batch_number
#SBATCH --partition=allcpu
