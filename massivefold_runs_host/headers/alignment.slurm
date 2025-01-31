#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --output=${cwd}/${logs_dir}/${sequence_name}/${run_name}/stdout-alignment-%x-%j.log
#SBATCH --error=${cwd}/${logs_dir}/${sequence_name}/${run_name}/stderr-alignment-%x-%j.log
#SBATCH --partition=allcpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread

source /etc/profile.d/modules.sh
source /gpfs/cssb/software/envrc


#--output=/gpfs/cssb/group/cssb-topf/natan/tests/test_massive_fold/MassiveFold/massivefold_runs_host/stdout-alignment-%x-%j.log
#--error=/gpfs/cssb/group/cssb-topf/natan/tests/test_massive_fold/MassiveFold/massivefold_runs_host/stderr-alignment-%x-%j.log
