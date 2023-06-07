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
export MPICH_GPU_SUPPORT_ENABLED=1

srun --job-name="test_gb" \
    --ntasks=1 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
    --gpus-per-task=1 \
    --constraint=MI250 \
    --account=rad4268 \
    --output="/lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/test_log_.out" \
    --error="/lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/test_errors_.err" \
    --gpu-bind=closest -- distribution_strategy=off -- \
    --time=24:00:00 \
    python /lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/DECON/bench_refset.py "/lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/data/Eucalyptus_Density_1654_Chaix_RMSE0.03" "0"