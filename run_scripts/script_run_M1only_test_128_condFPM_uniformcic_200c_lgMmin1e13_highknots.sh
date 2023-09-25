#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=run_M1only_HR_highknots
#SBATCH -p gpu
#SBATCH -C a100,ib
#SBATCH --gpus=1
#SBATCH --output=/mnt/home/spandey/ceph/AR_NPE/run_scripts/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/AR_NPE/run_scripts/slurm_logs/%x.%j.err

module load python
source /mnt/home/spandey/ceph/.venv/bin/activate

export OMP_NUM_THREADS=4

cd /mnt/home/spandey/ceph/AR_NPE/
time srun python /mnt/home/spandey/ceph/AR_NPE/notebooks/TEST_ROCKSTAR_RUNS/FINAL_CHECK_M1_only.py
echo "done"


