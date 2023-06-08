# Set the root folder path
root_folder="/lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/data"

# Set the output folder path
output_folder="/lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/results"

module load tensorflow-gpu/py3

rm -rf /lus/home/CT3/rad4268/dcros/.cache/miopen
export MIOPEN_LOG_LEVEL=3
export MIOPEN_USER_DB_PATH="/tmp"

for folder in "$root_folder"/*; do
  if [ -d "$folder" ]; then
    folder_name=$(basename "$folder")
    echo "Launching job for folder: $folder_name"

    # Create a separate output folder for each job
    job_output_folder="$output_folder/$folder_name"
    dataset_folder="$root_folder/$folder_name"
    mkdir -p "$job_output_folder"


    # Create a script file for the job
    run_name="${folder_name}"
    script_file="job_${run_name}.sh"
    echo "#!/bin/bash" > "$script_file"
    echo "#SBATCH --account=rad4268" >> "$script_file"
    echo "#SBATCH --constraint=MI250" >> "$script_file"
    echo "#SBATCH --nodes=1" >> "$script_file"
    echo "#SBATCH --ntasks-per-node=1 --cpus-per-task=4 --gpus-per-task=1" >> "$script_file"
    echo "#SBATCH --time=24:00:00" >> "$script_file"
    echo "" >> "$script_file"

    echo "python /lus/scratch/CT3/rad4268/dcros/gbeurier/DECON/Bench_onefile.py \"$dataset_folder\"" >> "$script_file"
    chmod +x "$script_file"

    # Launch the job using srun
    srun --job-name="${run_name}" \
          --ntasks=1 \
          --ntasks-per-node=1 \
          --cpus-per-task=4 \
          --gpus-per-task=1 \
          --constraint=MI250 \
          --account=rad4268 \
          --output="/lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/${run_name}_log_.out" \
          --error="/lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/${run_name}_errors_.err" \
          --time=24:00:00 \
          "./$script_file" &
  fi
  # break
done
