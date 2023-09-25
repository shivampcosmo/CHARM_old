#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=run_test_128_condFPM_200c_lgMmin5e13_vhres
#SBATCH -p gpu
#SBATCH -C h100,ib
#SBATCH --gpus=1
#SBATCH --output=/mnt/home/spandey/ceph/AR_NPE/run_scripts/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/AR_NPE/run_scripts/slurm_logs/%x.%j.err

module load python
source /mnt/home/spandey/ceph/.venv/bin/activate

export OMP_NUM_THREADS=4

cd /mnt/home/spandey/ceph/AR_NPE/
time srun python /mnt/ceph/users/spandey/AR_NPE/nf/run_final.py run_test_128_condFPM_uniformcic_200c_lgMmin5e13_vhres.yaml
echo "done"


