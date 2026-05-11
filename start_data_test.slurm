#!/usr/bin/env bash
#SBATCH --job-name=BA_test_run       
#SBATCH --partition=epyc-gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jonas02boehringer@gmail.com

set -e  # sofort abbrechen bei Fehlern

# -----------------------------
# 4. Umgebung aktivieren
# -----------------------------
source venv/bin/activate

# -----------------------------
# 5. Threading kontrollieren
# -----------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# -----------------------------
# 8. Start
# -----------------------------
srun python src/base_model.py #--path /hpc/gpfs2/home/u/boehrijo/final_runs/run1/clean/test/CPAP-014-3