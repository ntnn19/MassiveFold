#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --output=${logs_dir}/${sequence_name}/${run_name}/alignment_%j.log
#SBATCH --error=${logs_dir}/${sequence_name}/${run_name}/alignment_%j.log
#SBATCH --partition=topfgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread

