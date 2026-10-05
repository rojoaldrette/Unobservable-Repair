#!/bin/bash
# Plantilla genérica de SLURM para el Monte Carlo.  Ajustar partición, módulos y
# entorno de Python a la supercomputadora (no la conozco todavía).
#
# Paso 1 (una vez por diseño):   sbatch --export=STEP=solve slurm_mc.sh
# Paso 2 (réplicas en array):    sbatch --array=0-24 --export=STEP=reps slurm_mc.sh
#                                (25 tareas x 10 réplicas = 250 = mc_replic)
#SBATCH --job-name=mc_reparar
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

# module load python/3.x            # <- ajustar
# source ~/venvs/tesis/bin/activate # <- ajustar (jax, jaxopt, scipy, pandas)

export PYTHONIOENCODING=utf-8
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"

DESIGN="--N 20000 --T 13 --spread 0.3 --spec flexible --outdir output"
BLOCK=10

cd "$(dirname "$0")"
mkdir -p logs output

if [ "$STEP" = "solve" ]; then
    python main.py --solve-only $DESIGN
else
    A=$((SLURM_ARRAY_TASK_ID * BLOCK))
    B=$((A + BLOCK))
    python main.py --reps ${A}:${B} $DESIGN -v
fi
