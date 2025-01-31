date
sequence_name=$sequence_name
run_name=$run_name
cwd=$cwd

cd $$cwd


#Determine number of GPUs on the node we ended up on.

JOB_LIST="${logs_dir}/${sequence_name}/${run_name}/$${sequence_name}_$${run_name}_jobarray_as_joblist.txt"
NUM_GPUS=$$(nvidia-smi -L | wc -l)
joblist_without_ext="$${JOB_LIST%.*}"
echo "# GPUs = $${NUM_GPUS}"
# Extract NUM_GPUS jobs from the list (handling concurrency)
LOCKFILE="$${joblist_without_ext}.lock"
exec 9>$$LOCKFILE
flock -x 9   # Lock the file to prevent race conditions

if [ ! -s "$$JOB_LIST" ]; then
    echo "No more jobs left. Exiting."
    flock -u 9  # Unlock before exiting
    exit 0
fi

head -n $$NUM_GPUS $$JOB_LIST > "$${joblist_without_ext}_$${SLURM_ARRAY_TASK_ID}.txt"  # Get k tasks
sed -i "1,$${NUM_GPUS}d" $$JOB_LIST
flock -u 9

< "$${joblist_without_ext}_$${SLURM_ARRAY_TASK_ID}.txt" parallel -j $$NUM_GPUS 'eval CUDA_VISIBLE_DEVICES=$$(({%} - 1)) {}' 2>&1 | tee -a $${SLURM_STDOUT}

date
