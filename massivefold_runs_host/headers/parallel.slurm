#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/cssb/group/cssb-topf/natan/tests/test_massive_fold/MassiveFold/massivefold_runs_host/run/stdout-%x-%j.log
#SBATCH --error=/gpfs/cssb/group/cssb-topf/natan/tests/test_massive_fold/MassiveFold/massivefold_runs_host/run/stderr-%x-%j.log
#SBATCH --partition=topfgpu
#SBATCH --nodes=1
#SBATCH --array=0-29

source /etc/profile.d/modules.sh
source /gpfs/cssb/software/envrc
