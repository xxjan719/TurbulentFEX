#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=QIDIFEX
#SBATCH --output=QIDIFEX_SAMPLE20000.out
#SBATCH --error=QIDIFEX_SAMPLE20000.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xingjianxu@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1                # Reduce number of tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=5gb               # Reduced memory usage 
# Removed GPU-specific options
#SBATCH --gres=gpu:1     # ✅ Generic GPU request (L4-compatible)
# Note that these reductions are just examples. You should tailor them to fit within your system's limits.

echo "===== SLURM JOB STARTED ====="
echo "Date      = $(date)"
echo "Host      = $(hostname -s)"
echo "Start Dir = $(pwd)"

module load conda
conda activate xxjan

# ✅ Run your Python script
python -u small_test.py --SAMPLE 5000

echo "===== JOB FINISHED ====="
date