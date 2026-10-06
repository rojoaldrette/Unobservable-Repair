# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/main.py
# Goal:           Punto de entrada del Monte Carlo (local o en la supercomputadora)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
Uso (desde claude/scripts/modelo_fin/):

  # 1) Resolver y guardar los equilibrios de los T regímenes (una vez por diseño)
  python main.py --solve-only --T 13 --spread 0.3

  # 2) Réplicas por bloques (p. ej. un array de SLURM con 25 tareas de 10 réplicas)
  python main.py --reps 0:10 --N 20000 --T 13 --spread 0.3
  python main.py --reps $((SLURM_ARRAY_TASK_ID*10)):$((SLURM_ARRAY_TASK_ID*10+10)) ...

  # Prueba de humo (minutos, en laptop): grid chico, 2 regímenes, pocos hogares
  python main.py --smoke

Cada bloque escribe output/mc_<diseño>_reps<a>-<b>.csv.  Para juntar:
  pandas.concat(map(pandas.read_csv, glob("output/mc_*.csv")))
'''

import argparse
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from calibracion import calibracion, NOMBRES
from montecarlo import MCDesign, prepare, montecarlo


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0:1", help="rango a:b de réplicas (semillas)")
    ap.add_argument("--N", type=int, default=20_000)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--T", type=int, default=13)
    ap.add_argument("--spread", type=float, default=0.3)
    ap.add_argument("--spec", default="flexible", choices=["flexible", "logit_R"])
    ap.add_argument("--n_s", type=int, default=None)
    ap.add_argument("--calib", default="tesis", choices=NOMBRES)
    ap.add_argument("--a_max", type=int, default=25)
    ap.add_argument("--se", action="store_true")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--solve-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse()
    if args.smoke:
        args.n_s, args.T, args.N, args.reps, args.verbose = 12, 2, 3_000, "0:1", True
        args.a_max = 7
        args.outdir = os.path.join(args.outdir, "smoke")
    g = calibracion(args.calib, a_max=args.a_max, n_s=args.n_s or 100)

    design = MCDesign(N=args.N, K=args.K, T=args.T, spread=args.spread, spec=args.spec, se=args.se)
    if args.solve_only:
        prepare(g, design, args.outdir, verbose=True)
    else:
        a, b = map(int, args.reps.split(":"))
        montecarlo(g, design, list(range(a, b)), args.outdir, verbose=args.verbose)
