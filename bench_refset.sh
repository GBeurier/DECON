#!/bin/bash
#SBATCH --account=rad4268
#SBATCH --constraint=MI250
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8 --cpus-per-task=8 --gpus-per-node=8
#SBATCH --threads-per-core=1 # --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.cros@cirad.fr
#SBATCH --time=24:00:00

# Set the root folder path
root_folder="/lus/home/PERSO/grp_dcros/dcros/gbeurier/data"

# Set the output folder path
output_folder="/lus/home/PERSO/grp_dcros/dcros/gbeurier"

# Load necessary modules
module load cuda

# Loop through each folder within the root folder
for folder in "$root_folder"/*; do
  if [ -d "$folder" ]; then
    folder_name=$(basename "$folder")
    echo "Launching job for folder: $folder_name"

    # Create a separate output folder for each job
    job_output_folder="$output_folder/$folder_name"
    mkdir -p "$job_output_folder"

    # Update job name, output, and error file names based on the folder being processed
    sbatch --job-name="job_$folder_name" \
           --output="log_$folder_name.%j.out" \
           --error="errors_$folder_name.%j.err" \
           --time=24:00:00 \
           --wrap="python /lus/home/PERSO/grp_dcros/dcros/gbeurier/bench_refset.py \"$folder\" \"$job_output_folder\""
  fi
done
