#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --output=${logs_dir}/${sequence_name}/${run_name}/post_treatment_%j.log
#SBATCH --error=${logs_dir}/${sequence_name}/${run_name}/post_treatment_%j.log
#SBATCH --partition=topfgpu
#SBATCH --nodes=1
#SBATCH --constraint="A100"

