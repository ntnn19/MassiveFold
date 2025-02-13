
# These environment variables are suggested by google deepmind at:
# https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md#additional-flags
# work around a known XLA issue causing the compilation time to greatly increase
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
# token < 5,120 on aingle A100 (80 GB) or H100 (80 GB)
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.95
# token > 5120
: '
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=true
export XLA_CLIENT_MEM_FRACTION=1.5
#export XLA_CLIENT_MEM_FRACTION=3.2
'
run_massivefold=$run_massivefold
sequence_name=$sequence_name
run_name=$run_name
data_dir=$data_dir
scripts_dir=$scripts_dir
max_template_date=$max_template_date

batch=$$SLURM_ARRAY_TASK_ID

cwd=$cwd
cd $cwd
json_batch_input=af3_batch_$${batch}.json
output_dir=$$(realpath ${output_dir})
output_path="$${output_dir}/$${sequence_name}/$${run_name}/"
echo output_path is $$output_path
mkdir -vp $${output_path}/
mf_container_path=$mf_container_path
af_container_path=$af_container_path
weights_dir=$weights_dir

path_to_msas=$${output_dir}/$${sequence_name}/$${run_name}/$${json_batch_input}
#jackhmmer_path=$$(singularity exec --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate mf-alphafold3 && which jackhmmer')
#hmmalign_path=$$(singularity exec --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate mf-alphafold3 && which hmmalign')
#hmmbuild_path=$$(singularity exec --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate mf-alphafold3 && which hmmbuild')
#hmmsearch_path=$$(singularity exec --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate mf-alphafold3 && which hmmsearch')
#echo "hmmsearch_path= $${hmmsearch_path}"
date

#echo "time singularity run --nv -B $${data_dir}:$${data_dir} -B $${weights_dir}:$${weights_dir} -B $${cwd}/af3_tmp:/tmp --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate mf-alphafold3 && run_alphafold.py --norun_data_pipeline --max_template_date $${max_template_date} --json_path $${path_to_msas} --db_dir $${data_dir} --model_dir $${weights_dir} --output_dir $${output_path} --jackhmmer_binary_path $${jackhmmer_path} --hmmalign_binary_path $${hmmalign_path} --hmmbuild_binary_path $${hmmbuild_path} --hmmsearch_binary_path $${hmmsearch_path}'" >> "${logs_dir}/${sequence_name}/${run_name}/$${sequence_name}_$${run_name}_jobarray_as_joblist.txt"
echo "time singularity run --nv -B $${data_dir}:$${data_dir} -B $${weights_dir}:$${weights_dir} -B $${cwd}/af3_tmp:/tmp --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs $${af_container_path} python $${run_massivefold} --norun_data_pipeline --max_template_date $${max_template_date} --json_path $${path_to_msas} --db_dir $${data_dir} --model_dir $${weights_dir} --output_dir $${output_path}" >> "${logs_dir}/${sequence_name}/${run_name}/$${sequence_name}_$${run_name}_jobarray_as_joblist.txt"

