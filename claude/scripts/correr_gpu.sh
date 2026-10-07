#!/bin/bash
# _____________________________________________________________________________
#
# Project:  Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:   claude/scripts/correr_gpu.sh
# Goal:     Correr todo en la supercomputadora (2 GPUs, sin SLURM), por fases
#
# Uso (desde cualquier carpeta; ver docs/manual.md, sección 6):
#
#   bash claude/scripts/correr_gpu.sh <fase>
#
#   prueba    pruebas + humo + una corrida chica (T = 3, N = 5,000) para medir tiempos
#   fase1     GPU 0: verdad + réplica 0 de modelo_fin (+ panel)
#             GPU 1: Gillingham sobre sus datos (MC completo) y compare.py
#             al final: cruce de la réplica 0 y un primer análisis
#   fase2     MC de modelo_fin, réplicas 1..REPS-1 repartidas en las dos GPUs
#   cruce     Gillingham sobre el panel de cada réplica (MC del cruce)
#   analisis  gráficas y tablas con todo lo que haya
#   todo      fase1 + fase2 + cruce + analisis, en orden
#   estado    GPUs, procesos vivos y cuántas réplicas hay terminadas
#
# Cada fase espera a que terminen sus procesos.  Para que siga corriendo al cerrar ssh,
# lanzarlo dentro de tmux (recomendado) o con nohup:
#   nohup bash claude/scripts/correr_gpu.sh todo > claude/output/logs/todo.log 2>&1 &
#
# El diseño se cambia con variables de entorno, sin editar el archivo:
#   CALIB=tesis_v0 A_MAX=7 REPS=50 bash claude/scripts/correr_gpu.sh fase1
# Las mismas variables tienen que usarse en TODAS las fases de un mismo diseño.
# _____________________________________________________________________________

set -uo pipefail

# Diseño (se puede sobrescribir desde fuera) ____________________________________
CALIB=${CALIB:-tesis}
A_MAX=${A_MAX:-25}
N_S=${N_S:-100}
T_REG=${T_REG:-13}
SPREAD=${SPREAD:-0.3}
N_HOG=${N_HOG:-20000}
K_ANIOS=${K_ANIOS:-2}
TIPOS=${TIPOS:-low_couple_poor,low_single_poor}
REPS=${REPS:-100}                    # réplicas de modelo_fin: 0..REPS-1
GILL_REPS=${GILL_REPS:-100}          # réplicas de Gillingham sobre sus datos
GILL_K=${GILL_K:-10}                 # años por hogar en Gillingham (como el paper)
PROC_POR_GPU=${PROC_POR_GPU:-1}      # procesos de modelo_fin por GPU en fase2 (1 o 2)
EXTRA=${EXTRA:-}                     # opciones extra para modelo_fin/estimar.py (p. ej. "--fix u_s")
VENV=${VENV:-$HOME/venvs/tesis}      # entorno virtual con jax[cuda12]

TAG_MF=${TAG_MF:-${CALIB}_A${A_MAX}_S${N_S}_T${T_REG}_N${N_HOG}_K${K_ANIOS}}
TAG_GILL=${TAG_GILL:-gill_propios_A${A_MAX}_N${N_HOG}_K${GILL_K}}
TAG_CRUCE=${TAG_CRUCE:-gill_cruce_${TAG_MF}}

# Rutas y entorno ______________________________________________________________
SCR="$(cd "$(dirname "$0")" && pwd)"              # claude/scripts
EST="$SCR/../output/estimaciones"
LOGS="$SCR/../output/logs/$TAG_MF"
mkdir -p "$LOGS" "$EST"

if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false        # varios procesos por GPU

DISENO="--calib $CALIB --a_max $A_MAX --n_s $N_S --T $T_REG --spread $SPREAD --N $N_HOG --K $K_ANIOS --types $TIPOS"
MF_DIR="$EST/modelo_fin/$TAG_MF"

info() { echo "[$(date '+%F %T')] $*"; }

# Espera a los procesos dados y cuenta cuántos fallaron
esperar() {
    local fallas=0 p
    for p in "$@"; do
        wait "$p" || fallas=$((fallas + 1))
    done
    if [ "$fallas" -gt 0 ]; then
        info "OJO: $fallas proceso(s) terminaron con error; revisar los logs en $LOGS"
    fi
    return "$fallas"
}

# Fases ________________________________________________________________________

prueba() {
    info "pruebas en la GPU 0 (log: $LOGS/prueba.log)"
    {
        cd "$SCR/modelo_fin" &&
        CUDA_VISIBLE_DEVICES=0 python -u -c "import jax; print(jax.devices())" &&
        CUDA_VISIBLE_DEVICES=0 python -u tests.py --n_s 12 --estructural &&
        CUDA_VISIBLE_DEVICES=0 python -u estimar.py --smoke &&
        cd "$SCR/gillingham" &&
        CUDA_VISIBLE_DEVICES=0 python -u estimar.py --smoke &&
        cd "$SCR/modelo_fin" &&
        CUDA_VISIBLE_DEVICES=0 python -u estimar.py --reps 0:1 --calib "$CALIB" --a_max "$A_MAX" \
            --n_s "$N_S" --T 3 --N 5000 --K "$K_ANIOS" --types "$TIPOS" $EXTRA \
            --tag "prueba_${TAG_MF}_T3" -v
    } 2>&1 | tee "$LOGS/prueba.log"
    info "tiempos de la corrida chica (columna 'segundos'):"
    cat "$EST/modelo_fin/prueba_${TAG_MF}_T3/resumen_reps0-0.csv" 2>/dev/null | cut -d, -f1-9
}

fase1() {
    info "fase 1: réplica 0 de modelo_fin (GPU 0) | Gillingham propio + compare (GPU 1)"
    cd "$SCR/modelo_fin"
    CUDA_VISIBLE_DEVICES=0 python -u estimar.py --reps 0:1 $DISENO $EXTRA --tag "$TAG_MF" \
        --guardar_panel -v > "$LOGS/mf_rep0.log" 2>&1 &
    local p_mf=$!

    (
        cd "$SCR/gillingham" &&
        CUDA_VISIBLE_DEVICES=1 python -u estimar.py --reps 0:"$GILL_REPS" --a_max "$A_MAX" \
            --types "$TIPOS" --N "$N_HOG" --K "$GILL_K" --tag "$TAG_GILL" -v \
            > "$LOGS/gill_propios.log" 2>&1
        cd "$SCR/comparacion" &&
        CUDA_VISIBLE_DEVICES=1 python -u compare.py --n_s "$N_S" > "$LOGS/compare.log" 2>&1
    ) &
    local p_gill=$!

    info "logs: $LOGS/mf_rep0.log, gill_propios.log, compare.log"
    esperar "$p_mf"
    if [ ! -f "$MF_DIR/verdad.npz" ]; then
        info "no se creó $MF_DIR/verdad.npz: revisar $LOGS/mf_rep0.log"
        esperar "$p_gill"
        return 1
    fi

    info "cruce de la réplica 0 (GPU 0)"
    cruce_rep 0 0
    esperar "$p_gill"
    analisis "${TAG_MF}_rep0"
}

fase2() {
    if [ ! -f "$MF_DIR/verdad.npz" ]; then
        info "falta $MF_DIR/verdad.npz: correr primero la fase1"
        return 1
    fi
    local nb=$((2 * PROC_POR_GPU)) tot=$((REPS - 1)) i a b gpu pids=()
    info "fase 2: réplicas 1..$((REPS - 1)) en $nb bloques ($PROC_POR_GPU por GPU)"
    cd "$SCR/modelo_fin"
    for ((i = 0; i < nb; i++)); do
        a=$((1 + i * tot / nb))
        b=$((1 + (i + 1) * tot / nb))
        [ "$a" -ge "$b" ] && continue
        gpu=$((i % 2))
        CUDA_VISIBLE_DEVICES=$gpu python -u estimar.py --reps "$a:$b" $DISENO $EXTRA --tag "$TAG_MF" \
            --rep_reporte 0 --guardar_panel > "$LOGS/mf_${a}-$((b - 1)).log" 2>&1 &
        pids+=($!)
        info "  GPU $gpu: réplicas $a-$((b - 1)) (log: $LOGS/mf_${a}-$((b - 1)).log)"
    done
    esperar "${pids[@]}"
}

# Una réplica del cruce: cruce_rep <k> <gpu>
cruce_rep() {
    local k=$1 gpu=$2 panel="$MF_DIR/panel_rep$1.csv.gz"
    if [ ! -f "$panel" ]; then
        info "  sin panel de la réplica $k, se salta"
        return 0
    fi
    cd "$SCR/gillingham"
    CUDA_VISIBLE_DEVICES=$gpu python -u estimar.py --panel "$panel" --reps "$k:$((k + 1))" \
        --a_max "$A_MAX" --types "$TIPOS" --tag "$TAG_CRUCE" >> "$LOGS/cruce.log" 2>&1 \
        || info "  OJO: falló el cruce de la réplica $k (ver $LOGS/cruce.log)"
}

cruce() {
    # Secuencial (todos escriben al mismo CSV) y de la última a la 0, para que
    # precios.csv / ccps.csv / distribucion.csv queden con el panel 0
    info "cruce: réplicas $((REPS - 1))..0 en la GPU 0 (log: $LOGS/cruce.log)"
    local k
    for ((k = REPS - 1; k >= 0; k--)); do
        cruce_rep "$k" 0
    done
}

analisis() {
    local nombre=${1:-$TAG_MF} args=()
    [ -d "$MF_DIR" ] && args+=(--mf "$MF_DIR")
    [ -d "$EST/gillingham/$TAG_CRUCE" ] && args+=(--gill "$EST/gillingham/$TAG_CRUCE")
    [ -d "$EST/gillingham/$TAG_GILL" ] && args+=(--gill "$EST/gillingham/$TAG_GILL")
    info "análisis -> claude/output/analisis/$nombre/"
    cd "$SCR/analisis"
    python -u main.py "${args[@]}" --nombre "$nombre" --todas 2>&1 | tee "$LOGS/analisis.log"
}

estado() {
    nvidia-smi
    echo
    echo "procesos:"
    ps -eo pid,etime,args | grep -E "estimar.py|compare.py|main.py" | grep -v grep || echo "  ninguno"
    echo
    python - "$EST" <<'EOF'
import glob, os, sys
import pandas as pd
for d in sorted(glob.glob(os.path.join(sys.argv[1], "*", "*"))):
    files = glob.glob(os.path.join(d, "resumen_reps*.csv"))
    if files:
        df = pd.concat(map(pd.read_csv, files))
        print(f"{os.path.relpath(d, sys.argv[1])}: {df['rep'].nunique()} réplicas terminadas")
EOF
}

# Main _________________________________________________________________________

info "diseño: $DISENO $EXTRA | REPS=$REPS | tags: $TAG_MF, $TAG_GILL, $TAG_CRUCE"
case "${1:-}" in
    prueba)   prueba ;;
    fase1)    fase1 ;;
    fase2)    fase2 ;;
    cruce)    cruce ;;
    analisis) analisis ;;
    todo)     fase1; fase2; cruce; analisis ;;
    estado)   estado ;;
    *)        echo "uso: bash correr_gpu.sh {prueba|fase1|fase2|cruce|analisis|todo|estado}"; exit 1 ;;
esac
info "listo: $1"
