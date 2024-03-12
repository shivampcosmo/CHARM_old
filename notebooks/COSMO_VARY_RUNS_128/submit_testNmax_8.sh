#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=16:00:00
#SBATCH --job-name=Nmax8_varycosmo
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH -C a100
#SBATCH --gpus=1
#SBATCH --output=/mnt/home/spandey/ceph/AR_NPE/notebooks/TEST_ROCKSTAR_RUNS/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/AR_NPE/notebooks/TEST_ROCKSTAR_RUNS/slurm_logs/%x.%j.err

module load python
__conda_setup="$('/mnt/home/spandey/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/home/spandey/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/home/spandey/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/home/spandey/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate ili-sbi
# source /mnt/home/spandey/ceph/.venv/bin/activate

# export OMP_NUM_THREADS=4

cd /mnt/home/spandey/ceph/AR_NPE/notebooks/COSMO_VARY_RUNS_128/
time srun python /mnt/home/spandey/ceph/AR_NPE/notebooks/COSMO_VARY_RUNS_128/run_COND_FPM_vary_cosmo_random.py
echo "done"


