#!/bin/bash
# _____________________________________________________________________________
#
# Script:   claude/scripts/slurm_estimar.sh
# Goal:     Plantilla de SLURM (GPU) para las estimaciones y el análisis
#
# Pasos (STEP), en este orden.  Ver docs/reporte_estimacion.md.
#
#   1. mf       modelo_fin: verdad + réplica 0 (D0 y D1) + guarda el panel
#               sbatch --export=ALL,STEP=mf slurm_estimar.sh
#   2. mf_mc    modelo_fin: Monte Carlo en array (10 réplicas por tarea)
#               sbatch --array=1-9 --export=ALL,STEP=mf_mc slurm_estimar.sh
#   3. gill     Gillingham sobre sus propios datos (réplica 0)
#               sbatch --export=ALL,STEP=gill slurm_estimar.sh
#   4. cruce    Gillingham sobre el panel de modelo_fin (el sesgo de ignorar la reparación)
#               sbatch --export=ALL,STEP=cruce slurm_estimar.sh
#   5. analisis gráficas y tablas (CPU basta)
#               sbatch --export=ALL,STEP=analisis slurm_estimar.sh
#
# Ajustar partición, módulos y entorno a la supercomputadora (no la conozco todavía).
# El tag de modelo_fin (TAG_MF) tiene que coincidir con el que arma estimar.py; se fija
# aquí con --tag para no depender de eso.
# _____________________________________________________________________________
#SBATCH --job-name=estimar
#SBATCH --gres=gpu:1                 # <- ajustar (p. ej. gpu:a100:1)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out

# module load cuda/12.x python/3.x   # <- ajustar
# source ~/venvs/tesis/bin/activate  # <- jax[cuda12], jaxopt, scipy, pandas, matplotlib

export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # no reservar toda la memoria de la GPU de entrada

# Diseño (igual en todos los pasos) ____________________________________________
A_MAX=25           # el del paper; calibración "tesis" (modelo_fin/calibracion.py)
N_S=100
T_REG=13
N_HOG=20000
K_ANIOS=2
TIPOS=low_couple_poor,low_single_poor
TAG_MF=tesis_A${A_MAX}_S${N_S}_T${T_REG}_N${N_HOG}_K${K_ANIOS}
TAG_GILL=gill_propios_A${A_MAX}
TAG_CRUCE=gill_cruce_A${A_MAX}
BLOQUE=10

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../output/estimaciones"
mkdir -p "$HERE/logs"

case "$STEP" in
  mf)
    cd "$HERE/modelo_fin"
    python -u estimar.py --reps 0:1 --calib tesis --a_max $A_MAX --n_s $N_S --T $T_REG --N $N_HOG \
        --K $K_ANIOS --types $TIPOS --method krylov --guardar_panel --tag $TAG_MF -v
    ;;
  mf_mc)
    cd "$HERE/modelo_fin"
    A=$((SLURM_ARRAY_TASK_ID * BLOQUE)); B=$((A + BLOQUE))
    python -u estimar.py --reps ${A}:${B} --calib tesis --a_max $A_MAX --n_s $N_S --T $T_REG \
        --N $N_HOG --K $K_ANIOS --types $TIPOS --method krylov --tag $TAG_MF --rep_reporte 0
    ;;
  gill)
    cd "$HERE/gillingham"
    python -u estimar.py --reps 0:1 --a_max $A_MAX --types $TIPOS --N $N_HOG --K 10 --tag $TAG_GILL -v
    ;;
  cruce)
    cd "$HERE/gillingham"
    python -u estimar.py --panel "$OUT/modelo_fin/$TAG_MF/panel_rep0.csv.gz" --a_max $A_MAX \
        --types $TIPOS --tag $TAG_CRUCE -v
    ;;
  analisis)
    cd "$HERE/analisis"
    python -u main.py --mf "$OUT/modelo_fin/$TAG_MF" --gill "$OUT/gillingham/$TAG_CRUCE" \
        --gill "$OUT/gillingham/$TAG_GILL" --nombre $TAG_MF --todas
    ;;
  *)
    echo "STEP desconocido: $STEP (mf | mf_mc | gill | cruce | analisis)"; exit 1 ;;
esac
