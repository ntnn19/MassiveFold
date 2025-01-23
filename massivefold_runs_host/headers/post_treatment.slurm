#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/cssb/group/cssb-topf/natan/tests/test_r2na/run/stdout-%x-%j.log
#SBATCH --error=/gpfs/cssb/group/cssb-topf/natan/tests/test_r2na/run/stderr-%x-%j.log
#SBATCH --partition=topfgpu
#SBATCH --nodes=1
#SBATCH --constraint="A100"

source /etc/profile.d/modules.sh
source /gpfs/cssb/software/envrc
