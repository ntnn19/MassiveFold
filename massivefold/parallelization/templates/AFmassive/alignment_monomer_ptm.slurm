# variables to adapt: used for alignment
sequence_name=$sequence_name
run_name=$run_name
run_massivefold=$run_massivefold
output=$output_dir
fafile=${input_dir}/$${sequence_name}.fasta
data_dir=$data_dir
max_template_date=$max_template_date
#max_template_date=2023-07-31
uniref_database=$uniref_database
container_path=$container_path
weights_dir=$weights_dir
cwd=$cwd

if [ -z "$$uniref_database" ]; then
  echo "No specified path for uniref30, use default ${data_dir}/uniref30/UniRef30_2021_03"
  uniref_database=$${data_dir}/uniref30/UniRef30_2021_03
fi

model_preset=monomer_ptm
alignments_only=true
db_preset=full_dbs
bfd_max_hits=100000
mgnify_max_hits=501
uniprot_max_hits=50000
uniref_max_hits=10000
use_gpu_relax=false

echo "singularity run --nv --no-home --pwd /app/massivefold_runs -B $${uniref_database}:$${uniref_database} -B $${data_dir}:$${data_dir} -B $${weights_dir}:$${weights_dir} -B $${cwd}:/app/massivefold_runs $${container_path} bash -c 'source /usr/local/etc/profile.d/conda.sh && conda activate massivefold && $${run_massivefold}
    --alignments_only=$${alignments_only} 
    --fasta_paths=$${fafile} 
    --output_dir=$${output} 
    --data_dir=$${weights_dir} 
    --db_preset=$${db_preset} 
    --use_gpu_relax=$${use_gpu_relax} 
    --model_preset=$${model_preset} 
    --max_template_date=$${max_template_date} 
    --bfd_max_hits=$${bfd_max_hits} 
    --mgnify_max_hits=$${mgnify_max_hits} 
    --uniprot_max_hits=$${uniprot_max_hits} 
    --uniref_max_hits=$${uniref_max_hits} 
    --uniref90_database_path=$${data_dir}/uniref90/uniref90.fasta 
    --mgnify_database_path=$${data_dir}/mgnify/mgy_clusters_2022_05.fa 
    --template_mmcif_dir=$${data_dir}/pdb_mmcif/mmcif_files
    --obsolete_pdbs_path=$${data_dir}/pdb_mmcif/obsolete.dat 
    --bfd_database_path=$${data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt 
    --pdb70_database_path=$${data_dir}/pdb70/pdb70
    --uniref30_database_path=$${uniref_database}/UniRef30_2023_02'"

time singularity run --nv --no-home --pwd /app/massivefold_runs -B $${uniref_database}:$${uniref_database} -B $${data_dir}:$${data_dir} -B $${weights_dir}:$${weights_dir} -B $${cwd}:/app/massivefold_runs $${container_path} bash -c "source /usr/local/etc/profile.d/conda.sh && conda activate massivefold && $${run_massivefold} \
    --alignments_only=$${alignments_only} \
    --fasta_paths=$${fafile} \
    --output_dir=$${output} \
    --data_dir=$${weights_dir} \
    --db_preset=$${db_preset} \
    --use_gpu_relax=$${use_gpu_relax} \
    --model_preset=$${model_preset} \
    --max_template_date=$${max_template_date} \
    --bfd_max_hits=$${bfd_max_hits} \
    --mgnify_max_hits=$${mgnify_max_hits} \
    --uniprot_max_hits=$${uniprot_max_hits} \
    --uniref_max_hits=$${uniref_max_hits} \
    --uniref90_database_path=$${data_dir}/uniref90/uniref90.fasta \
    --mgnify_database_path=$${data_dir}/mgnify/mgy_clusters_2022_05.fa \
    --template_mmcif_dir=$${data_dir}/pdb_mmcif/mmcif_files \
    --obsolete_pdbs_path=$${data_dir}/pdb_mmcif/obsolete.dat \
    --bfd_database_path=$${data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --pdb70_database_path=$${data_dir}/pdb70/pdb70 \
    --uniref30_database_path=$${uniref_database}/UniRef30_2023_02

date
