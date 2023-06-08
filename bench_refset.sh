# Set the root folder path
root_folder="/lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/data"

# Set the output folder path
output_folder="/lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/results"

module load tensorflow-gpu/py3
# source "/lus/home/PERSO/grp_dcros/dcros/deeplearning_oilpalm/bin/activate"


# directory="/lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/3rd"
# # Iterate over .whl files
# for file in "$directory"/*.whl; do
#     echo "Installing $file"
#     pip install --no-deps "$file"
# done
num_iterations=6
for ((index=0; index < num_iterations; index++)); do
  # Loop through each folder within the root folder
  for folder in "$root_folder"/*; do
    if [ -d "$folder" ]; then
      folder_name=$(basename "$folder")
      echo "Launching job for folder: $folder_name"

      # Create a separate output folder for each job
      job_output_folder="$output_folder/$folder_name"
      dataset_folder="$root_folder/$folder_name"
      mkdir -p "$job_output_folder"


      # Create a script file for the job
      run_name="${index}_${folder_name}"
      script_file="job_${run_name}.sh"
      echo "#!/bin/bash" > "$script_file"
      echo "#SBATCH --account=rad4268" >> "$script_file"
      echo "#SBATCH --constraint=MI250" >> "$script_file"
      echo "#SBATCH --nodes=1" >> "$script_file"
      echo "#SBATCH --ntasks-per-node=8 --cpus-per-task=2" >> "$script_file"
      echo "#SBATCH --time=24:00:00" >> "$script_file"
      echo "" >> "$script_file"

      echo "python /lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/DECON/bench_refset.py \"$dataset_folder\" \"$index\" " >> "$script_file"
      chmod +x "$script_file"

      # Launch the job using srun
      srun --job-name="${run_name}" \
            --ntasks=1 \
            --cpus-per-task=2 \
            --constraint=MI250 \
            --account=rad4268 \
            --output="/lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/${run_name}_log_.out" \
            --error="/lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/${run_name}_errors_.err" \
            --time=24:00:00 \
            "./$script_file" &
    fi
    # break
  done
  sleep 10
  # break
done
          # --gpus-per-task=2 \
          # --gres=gpu:2 \
