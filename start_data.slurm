#!/usr/bin/env bash
#SBATCH --job-name=BA_test_run       
#SBATCH --partition=epyc-gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:3
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jonas02boehringer@gmail.com

set -e  # sofort abbrechen bei Fehlern

# -----------------------------
# 1. Setup Run Directory (FREEZE)
# -----------------------------
RUN_DIR=~/ba_runs/run_${SLURM_JOB_ID}
mkdir -p $RUN_DIR

echo "Run dir: $RUN_DIR"

# Code snapshot
cp -r ~/ba $RUN_DIR/

# Optional: Version markieren
echo "VERSION: $(date)" > $RUN_DIR/version.txt

# -----------------------------
# 2. Wechsel in Snapshot
# -----------------------------
cd $RUN_DIR/ba

# -----------------------------
# 3. Python Cache entfernen (wichtig: im Snapshot!)
# -----------------------------
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete

# -----------------------------
# 4. Umgebung aktivieren
# -----------------------------
source venv/bin/activate

# -----------------------------
# 5. Threading kontrollieren
# -----------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# -----------------------------
# 7. Debug (optional aber empfohlen)
# -----------------------------
echo "Running from: $(pwd)"
python -c "import inspect; import src.data_processing.featureZone as f; print('FeatureZone file:', inspect.getfile(f.FeatureZone))"

# -----------------------------
# 8. Start
# -----------------------------
srun python src/data_processing/workflow.py --input_path ~/run3