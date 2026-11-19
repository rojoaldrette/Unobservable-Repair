#!/bin/bash
# _____________________________________________________________________________
#
# Project:  Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:   claude/niu/final/correr_gpu.sh
# Goal:     Correr en la workstation (2 GPUs) el diseño de docs/tex/tesis.tex
#
# Uso (desde cualquier carpeta; conviene dentro de tmux):
#
#   bash claude/niu/final/correr_gpu.sh pruebas    # tests + humo (minutos)
#   bash claude/niu/final/correr_gpu.sh verdad     # equilibrio verdadero (teoria/R_LB) + panel
#   bash claude/niu/final/correr_gpu.sh mc         # MC: réplicas 0..REPS-1 en las dos GPUs
#   bash claude/niu/final/correr_gpu.sh faltantes  # re-estima las (rep, diseño) que fallaron
#   bash claude/niu/final/correr_gpu.sh resumen    # resumen_mc.csv
#   bash claude/niu/final/correr_gpu.sh por_edad   # equilibrio en cada theta estimado
#   bash claude/niu/final/correr_gpu.sh estado     # GPUs, procesos y réplicas terminadas
#
# Revisar la salida de "verdad" ANTES de lanzar "mc": masa_borde_w < 0.005 y que haya
# reparación razonable en las dos marcas (heavy brown tiene R constante = 16.6).
#
# Variables: REPS (50), REP0 (0), GPUS (2), TAG (base).  Réplicas REP0..REPS-1.
# Instrucciones completas: claude/niu/final/instrucciones.md
# _____________________________________________________________________________

set -uo pipefail

REPS=${REPS:-50}      # réplicas REP0..REPS-1
REP0=${REP0:-0}
GPUS=${GPUS:-2}
TAG=${TAG:-base}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/modelo_tesis"
LOGS="$HERE/output/logs"
mkdir -p "$LOGS"
export PYTHONIOENCODING=utf-8

run_split() {
    # $1 = script, $2 = argumentos extra; reparte 0..REPS-1 entre las GPUs y espera
    local script=$1 extra=${2:-}
    local tot=$(( REPS - REP0 ))
    local per=$(( (tot + GPUS - 1) / GPUS ))
    for ((g = 0; g < GPUS; g++)); do
        local a=$(( REP0 + g * per )) b=$(( REP0 + (g + 1) * per ))
        (( b > REPS )) && b=$REPS
        (( a >= b )) && continue
        echo "GPU $g: $script --reps $a:$b $extra"
        ( cd "$SRC" && CUDA_VISIBLE_DEVICES=$g python -u "$script" --reps "$a:$b" --tag "$TAG" $extra \
            > "$LOGS/${script%.py}_${a}-$((b - 1))${extra:+_extra}.log" 2>&1 ) &
    done
    wait
}

case "${1:-}" in
    pruebas)
        ( cd "$HERE/gillingham" && python -u tests.py ) && \
        ( cd "$SRC" && python -u tests.py ) && \
        ( cd "$SRC" && python -u montecarlo.py --smoke ) && \
        ( cd "$SRC" && python -u por_edad_mc.py --smoke )
        ;;
    verdad)
        ( cd "$SRC" && CUDA_VISIBLE_DEVICES=0 python -u teoria.py --tag R_LB --guardar_panel -v ) \
            2>&1 | tee "$LOGS/teoria.log"
        ;;
    mc)        run_split montecarlo.py ;;
    faltantes) run_split montecarlo.py --solo_faltantes ;;
    resumen)   ( cd "$SRC" && python -u montecarlo.py --summarize --tag "$TAG" ) ;;
    por_edad)
        # por_edad_mc.py se salta lo que ya está: una sola GPU, se puede cortar y volver a correr
        ( cd "$SRC" && CUDA_VISIBLE_DEVICES=0 python -u por_edad_mc.py --reps "0:$REPS" --tag "$TAG" ) \
            2>&1 | tee "$LOGS/por_edad.log"
        ;;
    estado)
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
        pgrep -af "montecarlo.py|por_edad_mc.py|teoria.py|estimar_panel.py" || echo "sin procesos"
        cat "$HERE"/output/modelo_tesis/montecarlo/"$TAG"/resumen_reps*.csv 2>/dev/null \
            | grep -c diseno | xargs echo "estimaciones terminadas (rep x diseño):"
        tail -qn 1 "$LOGS"/montecarlo_*.log 2>/dev/null
        df -h "$HERE/output" | tail -1
        ;;
    *)  sed -n 9,22p "${BASH_SOURCE[0]}" ;;
esac
