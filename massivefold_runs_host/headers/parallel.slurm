#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1            # Number of nodes
#SBATCH --hint=nomultithread # Disable hyperthreading
#SBATCH --error=${logs_dir}/${sequence_name}/${run_name}/stderr-parallel_%a-%x-%j.log
#SBATCH --output=${logs_dir}/${sequence_name}/${run_name}/stdout-parallel_%a-%x-%j.log
#SBATCH --array=0-999 # Should be good for most cases
#SBATCH --partition=topfgpu
