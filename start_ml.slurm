#!/usr/bin/env bash
#SBATCH --job-name=BA_test_run       
#SBATCH --partition=epyc-gpu-test                   # Partition mit GPU (falls verfügbar)
#SBATCH --nodes=1                                   # Anzahl Nodes
#SBATCH --ntasks=1                                  # Anzahl Tasks
#SBATCH --cpus-per-task=32                          # CPU Kerne pro Task
#SBATCH --gres=gpu:1                                # Anzahl GPUs (pro Node)
#SBATCH --mem=128G                                  # Arbeitsspeicher
#SBATCH --time=03:00:00                             # Max Laufzeit 6h (hh:mm:ss)
#SBATCH --mail-type=END,FAIL                        # Mail bei Ende oder Fehler
#SBATCH --mail-user=jonas02boehringer@gmail.com     # Empfänger

# -----------------------------
# Aktiviere virtuelle Umgebung
# -----------------------------
source ~/ba/venv/bin/activate          # Pfad zu deiner venv
# -----------------------------
# Limitiere OpenMP Threads
# -----------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# -----------------------------
# Starte den Workflow
# -----------------------------
srun python ~/ba/src/ml_training_validation/workflow.py --input_path ~/final_runs/run3/feature --output_path /hpc/gpfs2/home/u/boehrijo/final_runs/run3/output
