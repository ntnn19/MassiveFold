date

#variables to adapt
sequence_name=$sequence_name
run_name=$run_name
fafile=${input_dir}/$${sequence_name}.fasta
run_massivefold=$run_massivefold
data_dir=$data_dir
scripts_dir=$scripts_dir
max_template_date=$max_template_date
#max_template_date=2023-07-31
uniref_database=$uniref_database
container_path=$container_path
weights_dir=$weights_dir
cwd=$cwd
cd $$cwd
if [ -z "$$uniref_database" ]; then
  echo "No specified path for uniref30, use default ${data_dir}/uniref30/UniRef30_2021_03"
  uniref_database=$${data_dir}/uniref30/UniRef30_2021_03
fi

output_dir=$$(realpath ${output_dir})
output_path="$${output_dir}/$${sequence_name}/$${run_name}/batch_$${SLURM_ARRAY_TASK_ID}"
echo output_path is $$output_path
mkdir -vp $${output_path}/$${sequence_name}
ln -s $${output_dir}/$${sequence_name}/msas $${output_path}/$${sequence_name}

# fixed variables
use_precomputed_msas=true
alignments_only=false
use_gpu_relax=false
models_to_relax=none

# variable as MF_parallel parameters
model_preset=$model_preset
db_preset=$db_preset
dropout=$dropout
dropout_structure_module=$dropout_structure_module
dropout_rates_filename=$dropout_rates_filename
max_recycles=$max_recycles
early_stop_tolerance=$early_stop_tolerance
bfd_max_hits=$bfd_max_hits
mgnify_max_hits=$mgnify_max_hits
uniprot_max_hits=$uniprot_max_hits
uniref_max_hits=$uniref_max_hits
templates=$templates
min_score=$min_score
max_score=$max_score
stop_recycling_below=$stop_recycling_below

# parameters for each task in the job array
batch_vars=$$(singularity exec --bind $${cwd}:/app/massivefold_runs --no-home --pwd /app/massivefold_runs \
    --env SLURM_ARRAY_TASK_ID=$${SLURM_ARRAY_TASK_ID} \
    --env sequence_name=$${sequence_name} \
    --env run_name=$${run_name} \
    --env scripts_dir=$${scripts_dir} \
    $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate massivefold &&
    set -e  # Exit on errors for debugging
    echo "Running inside container..."

    # Check if environment variables are set correctly
    echo "sequence_name=$${sequence_name}"
    echo "run_name=$${run_name}"
    echo "SLURM_ARRAY_TASK_ID=$${SLURM_ARRAY_TASK_ID}"

    batches_path="${logs_dir}/${sequence_name}/${run_name}/$${sequence_name}_$${run_name}_batches.json"
    echo "batches_path=$${batches_path}"

    batch_model=$$($${scripts_dir}/get_batch.py --batch_id $$SLURM_ARRAY_TASK_ID --json_path $${batches_path} --element model || echo "ERROR")
    batch_start=$$($${scripts_dir}/get_batch.py --batch_id $$SLURM_ARRAY_TASK_ID --json_path $${batches_path} --element start || echo "ERROR")
    batch_end=$$($${scripts_dir}/get_batch.py --batch_id $$SLURM_ARRAY_TASK_ID --json_path $${batches_path} --element end || echo "ERROR")

    echo "batch_model=$${batch_model}"
    echo "batch_start=$${batch_start}"
    echo "batch_end=$${batch_end}"
' | tee "${logs_dir}/${sequence_name}/${run_name}/batch_variables_$$SLURM_ARRAY_TASK_ID.txt")  # Save output to a file

# Read values from the file to use them in the host environment
batch_model=$$(awk -F= '/batch_model/ {print $$2}' $logs_dir/$sequence_name/$run_name/batch_variables_$$SLURM_ARRAY_TASK_ID.txt)
batch_start=$$(awk -F= '/batch_start/ {print $$2}' $logs_dir/$sequence_name/$run_name/batch_variables_$$SLURM_ARRAY_TASK_ID.txt)
batch_end=$$(awk -F= '/batch_end/ {print $$2}' $logs_dir/$sequence_name/$run_name/batch_variables_$$SLURM_ARRAY_TASK_ID.txt)

echo Model used is $$batch_model, batch computed between prediction $$batch_start and $$batch_end
echo TASK_ID $$SLURM_ARRAY_TASK_ID json_path $${batches_path}

echo "time singularity run --nv --no-home --pwd /app/massivefold_runs -B $${uniref_database}:$${uniref_database} -B $${data_dir}:$${data_dir} -B $${weights_dir}:$${weights_dir} -B $${cwd}:/app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate massivefold && $${run_massivefold} --fasta_paths=$${fafile} --output_dir=$${output_path} --data_dir=$${weights_dir} --db_preset=$${db_preset} --model_preset=$${model_preset} --models_to_relax=$${models_to_relax} --use_gpu_relax=$${use_gpu_relax} --max_template_date=$${max_template_date} --use_precomputed_msas=$${use_precomputed_msas} --end_prediction=$${batch_end} --alignments_only=$${alignments_only} --dropout=$${dropout} --dropout_structure_module=$${dropout_structure_module} --dropout_rates_filename=$${dropout_rates_filename} --max_recycles=$${max_recycles} --early_stop_tolerance=$${early_stop_tolerance} --min_score=$${min_score} --max_score=$${max_score} --stop_recycling_below=$${stop_recycling_below} --bfd_max_hits=$${bfd_max_hits} --mgnify_max_hits=$${mgnify_max_hits} --uniprot_max_hits=$${uniprot_max_hits} --uniref_max_hits=$${uniref_max_hits} --models_to_use=$${batch_model} --start_prediction=$${batch_start} --templates=$${templates} --uniref90_database_path=$${data_dir}/uniref90/uniref90.fasta --mgnify_database_path=$${data_dir}/mgnify/mgy_clusters_2022_05.fa --template_mmcif_dir=$${data_dir}/pdb_mmcif/mmcif_files --obsolete_pdbs_path=$${data_dir}/pdb_mmcif/obsolete.dat --bfd_database_path=$${data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt --pdb_seqres_database_path=$${data_dir}/pdb_seqres/pdb_seqres.txt --uniref30_database_path=$${uniref_database}/UniRef30_2023_02 --uniprot_database_path=$${data_dir}/uniprot/uniprot.fasta'" >> "${logs_dir}/${sequence_name}/${run_name}/$${sequence_name}_$${run_name}_jobarray_as_joblist.txt"


date
