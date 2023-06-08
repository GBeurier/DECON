#!/bin/bash
#SBATCH --account=rad4268
#SBATCH --constraint=MI250
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8 --cpus-per-task=8 --gpus-per-node=8
#SBATCH --threads-per-core=1 # --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=beurier@cirad.fr
#SBATCH --time=24:00:00


module purge
module load tensorflow-gpu/py3

srun --job-name="TF_crash" \
    --ntasks=1 \
    --ntasks-per-node=4 \
    --cpus-per-task=4 \
    --gpus-per-task=1 \
    --constraint=MI250 \
    --account=rad4268 \
    --output="/lus/home/PERSO/grp_dcros/dcros/scratch/log_tensorflow_crash.out" \
    --error="/lus/home/PERSO/grp_dcros/dcros/scratch/errors_tensorflow_crash.err" \
    --time=24:00:00 \
    python /lus/home/PERSO/grp_dcros/dcros/scratch/_run_crash.py