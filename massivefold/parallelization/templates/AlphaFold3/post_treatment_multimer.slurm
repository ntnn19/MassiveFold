sequence_name=$sequence_name
run_name=$run_name
output=${output_dir}/$${sequence_name}/$${run_name}/

scripts_dir=$scripts_dir
batches_file=${logs_dir}/$${sequence_name}/$${run_name}/$${sequence_name}_$${run_name}_batches.json
pkl_format=$pkl_format
container_path=$container_path
cwd=$cwd
mf_plots_path=$$(singularity exec --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate massivefold && which $run_massivefold_plots)
date
# adapt colabfold output for massivefold
echo "$${scripts_dir}/unifier.py
  --batches_file $${batches_file}
  --conversion output
  --to_convert $$output
  --tool AlphaFold3"

echo "time singularity exec --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs
        --env sequence_name=$${sequence_name}
        --env run_name=$${run_name}
        --env scripts_dir=$${scripts_dir}
        $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate massivefold &&
        $${scripts_dir}/unifier.py --batches_file $$batches_file --conversion output --to_convert $$output --tool AlphaFold3'"

time singularity exec --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs \
        --env sequence_name=$${sequence_name} \
        --env run_name=$${run_name} \
        --env scripts_dir=$${scripts_dir} \
        $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate massivefold && \
        $${scripts_dir}/unifier.py --batches_file $$batches_file --conversion output --to_convert $$output --tool AlphaFold3'

# rename and move files
echo "time singularity run --nv --no-home --pwd /app/massivefold_runs -B $${cwd}:/app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate massivefold && $${scripts_dir}/organize_outputs.py --batches_path $$output'"
time singularity run --nv --no-home --pwd /app/massivefold_runs -B $${cwd}:/app/massivefold_runs $${container_path} bash -c "source /usr/local/etc/profile.d/conda.sh && conda activate massivefold && $${scripts_dir}/organize_outputs.py --batches_path $$output"


# plots
echo "Generating plots"

echo "singularity run --nv --no-home --pwd /app/massivefold_runs -B $${cwd}:/app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate massivefold && python3 $${mf_plots_path}
        --input_path=$${output}
        --top_n_predictions=$MF_plots_top_n_predictions
        --chosen_plots=$MF_plots_chosen_plots'"

time singularity run --nv --no-home --pwd /app/massivefold_runs -B $${cwd}:/app/massivefold_runs $${container_path} bash -c "source /usr/local/etc/profile.d/conda.sh && conda activate massivefold && python3 $${mf_plots_path} \
        --input_path=$${output} \
        --top_n_predictions=$MF_plots_top_n_predictions \
        --chosen_plots=$MF_plots_chosen_plots"

date
