singularity run --nv --no-home --env SLURM_CONF=/etc/slurm/slurm.conf -B /etc/slurm:/etc/slurm -B /run/munge:/run/munge -B /etc/passwd:/etc/passwd --pwd /app/massivefold_runs -B $(pwd):/app/massivefold_runs -B /gpfs/cssb/group/cssb-topf/natan/tests/test_massive_fold/MassiveFold/massivefold/parallelization/templates/AFmassive/alignment_multimer.slurm:/app/massivefold/parallelization/templates/AFmassive/alignment_multimer.slurm -B /gpfs/cssb/group/cssb-topf/natan/tests/test_massive_fold/MassiveFold/massivefold/parallelization/templates/AFmassive/jobarray_multimer.slurm:/app/massivefold/parallelization/templates/AFmassive/jobarray_multimer.slurm ../../../../singularity_containers/massive_fold/massive_fold_.sif  bash -c "source /usr/local/etc/profile.d/conda.sh && conda activate massivefold && ./run_massivefold.sh -s input/H1140.fasta -r afm_default_run -p 5 -f AFmassive_params.json -t AFmassive"