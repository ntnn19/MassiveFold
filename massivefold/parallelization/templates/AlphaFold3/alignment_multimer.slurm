
mf_following_msas="$${MF_FOLLOWING_MSAS}"

# variables to adapt: used for alignment>
run_massivefold=$run_massivefold
sequence_name=$sequence_name
run_name=$run_name
output=$output_dir
data_dir=$data_dir
logs_dir=$logs_dir
scripts_dir=$scripts_dir
container_path=$container_path
weights_dir=$weights_dir
cwd=$cwd
jackhmmer_path=$$(singularity exec --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate mf-alphafold3 && which jackhmmer)
hmmalign_path=$$(singularity exec --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate mf-alphafold3 && which hmmalign)
hmmbuild_path=$$(singularity exec --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate mf-alphafold3 && which hmmbuild)
hmmsearch_path=$$(singularity exec --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate mf-alphafold3 && which hmmsearch)
cd $$cwd
mkdir $${cwd}/af3_tmp
json_path=${input_dir}/alphafold3_json_requests/$${sequence_name}.json
json_path_dir=$$(dir $${json_path})
#source $$(conda info --base)/etc/profile.d/conda.sh
#conda activate mf-alphafold3

date
echo "singularity run --nv --no-home --pwd /app/massivefold_runs -B $${cwd}/af3_tmp:/tmp -B $${data_dir}:$${data_dir} -B $${weights_dir}:$${weights_dir} -B $${cwd}:/app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate mf-alphafold3 && $${run_massivefold}
  --norun_inference
  --json_path $${json_path}
  --db_dir $${data_dir}
  --model_dir $${weights_dir}
  --output_dir $${output}/$${sequence_name}
  --jackhmmer_binary_path $${jackhmmer_path}
  --hmmalign_binary_path $${hmmalign_path}
  --hmmbuild_binary_path $${hmmbuild_path}
  --hmmsearch_binary_path $${hmmsearch_path}
'"

time singularity run --nv --no-home --pwd /app/massivefold_runs -B $${cwd}/af3_tmp:/tmp -B $${data_dir}:$${data_dir} -B $${weights_dir}:$${weights_dir} -B $${cwd}:/app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate mf-alphafold3 && $${run_massivefold} \
  --norun_inference \
  --json_path $${json_path} \
  --db_dir $${data_dir} \which
  --model_dir $${weights_dir} \
  --output_dir $${output}/$${sequence_name} \
  --jackhmmer_binary_path $${jackhmmer_path} \
  --hmmalign_binary_path $${hmmalign_path} \
  --hmmbuild_binary_path $${hmmbuild_path} \
  --hmmsearch_binary_path $${hmmsearch_path}
'
date


if [[ -n $${mf_following_msas}  ]] && [[ $${mf_following_msas} == "true"  ]]; then
    singularity exec --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs \
        --env sequence_name=$${sequence_name} \
        --env run_name=$${run_name} \
        --env scripts_dir=$${scripts_dir} \
        $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate mf-alphafold3 && \
        set -e  # Exit on errors for debugging \
        echo "Running inside container..." \
        # Check if environment variables are set correctly
        echo "sequence_name=$${sequence_name}" \
        echo "run_name=$${run_name}" \
        batches_path="${logs_dir}/${sequence_name}/${run_name}/$${sequence_name}_$${run_name}_batches.json" \
        echo "batches_path=$${batches_path}" \
        echo "./$${scripts_dir}/unifier.py --conversion input_inference --batches_file $$batches_file --json_params $$json_params --to_convert $$json_msas --tool AlphaFold3" \
        $${scripts_dir}/unifier.py --conversion input_inference --batches_file $$batches_file --json_params $$json_params --to_convert $$json_msas --tool AlphaFold3'
else
  echo "Only MSAS are computed, no following inference."
fi


