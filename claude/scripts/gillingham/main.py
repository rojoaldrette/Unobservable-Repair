# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/main.py
# Goal:           Punto de entrada del Monte Carlo de Gillingham (local o SLURM)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           04/10/2026
#
# _____________________________________________________________________________
'''
Uso (desde claude/scripts/gillingham/):

  python main.py --smoke                         # prueba de humo (~1-2 min)
  python main.py --reps 0:50 --N 20000 --K 10 -v # 50 réplicas
  python main.py --reps 0:50 --fix tc_buy        # fija tc_buy en la verdad
  python main.py --summarize                     # junta los CSV y escribe el resumen

  # Dos tipos de hogar (Tablas 7-10), mitad y mitad
  python main.py --reps 0:50 --N 40000 --types low_couple_poor,low_single_poor -v

Salida: claude/output/montecarlo/gillingham/mc_<diseño>_reps<a>-<b>.csv y
        resumen_<diseño>.csv
'''

import argparse
import dataclasses
import glob
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import pandas as pd

from params import GParams, GTypes
from theta import FIELDS
from montecarlo import MCDesign, montecarlo, summarize

OUTDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "..", "output", "montecarlo", "gillingham"))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0:1", help="rango a:b de réplicas (semillas)")
    ap.add_argument("--N", type=int, default=20_000)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--a_max", type=int, default=25)
    ap.add_argument("--infos", default="parcial,completa")
    ap.add_argument("--fix", default="", help="parámetros fijos en la verdad, p. ej. tc_buy,mu")
    ap.add_argument("--types", default="low_couple_poor", help="tipos de PAPER_TYPES")
    ap.add_argument("--f", default="", help="fracciones de cada tipo (default: iguales)")
    ap.add_argument("--n_starts", type=int, default=2)
    ap.add_argument("--perturb", type=float, default=0.1)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse()
    if args.smoke:
        args.N, args.K, args.reps, args.n_starts, args.verbose = 5_000, 5, "0:2", 1, True
        args.outdir = os.path.join(args.outdir, "smoke")
    g = dataclasses.replace(GParams(), a_max=args.a_max)
    fix = set(filter(None, args.fix.split(",")))
    names = tuple(args.types.split(","))
    f = tuple(map(float, args.f.split(","))) if args.f else tuple([1.0 / len(names)] * len(names))
    design = MCDesign(N=args.N, K=args.K, infos=tuple(args.infos.split(",")),
                      free=tuple(k for k in FIELDS if k not in fix),
                      n_starts=args.n_starts, perturb=args.perturb, types=GTypes(names, f))

    if args.summarize:
        files = glob.glob(os.path.join(args.outdir, f"mc_{design.tag(g)}_reps*.csv"))
        df = pd.concat(map(pd.read_csv, files), ignore_index=True).drop_duplicates("rep")
        res = summarize(df, g, design)
        path = os.path.join(args.outdir, f"resumen_{design.tag(g)}.csv")
        res.to_csv(path, index=False)
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(f"{len(df)} réplicas\n", res.round(4).to_string(index=False))
        print("->", path)
    else:
        a, b = map(int, args.reps.split(":"))
        montecarlo(g, design, list(range(a, b)), args.outdir, verbose=args.verbose)
