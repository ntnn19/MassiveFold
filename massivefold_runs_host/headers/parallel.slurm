#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1            # Number of nodes
#SBATCH --partition=topfgpu
#SBATCH --array=0 # Should be good for most cases
#SBATCH --error=${cwd}/${logs_dir}/${sequence_name}/${run_name}/stderr-parallel_%a-%x-%j.log
#SBATCH --output=${cwd}/${logs_dir}/${sequence_name}/${run_name}/stdout-parallel_%a-%x-%j.log


source /etc/profile.d/modules.sh
source /gpfs/cssb/software/envrc
module load cuda
