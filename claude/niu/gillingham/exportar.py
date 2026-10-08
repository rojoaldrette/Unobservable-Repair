# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/gillingham/exportar.py
# Goal:           Resolver el equilibrio de Gillingham y exportarlo por (j, a) para comparar
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
    python -u exportar.py --outdir <carpeta> [--a_max 25 --brands ... --types ... --nocar ...]

Escribe en <carpeta>:
    gill_por_edad.csv    j, a, q, P, keep   (q y keep agregados sobre tipos con pesos f)
    gill_resumen.json    estadísticas de mercado (equilibrio.market_stats)
Lo usa modelo_tesis/teoria.py (como proceso aparte: los dos paquetes comparten nombres de módulo).
'''

import argparse
import json
import os

import numpy as np
import pandas as pd

from params import Config, theta
from utils import dims, split_states
from equilibrio import solve, equilibrium_objects, market_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--a_max", type=int, default=Config.a_max)
    ap.add_argument("--brands", default=",".join(Config.brands))
    ap.add_argument("--types", default=",".join(Config.types))
    ap.add_argument("--nocar", default=Config.nocar)
    args = ap.parse_args()
    types = tuple(args.types.split(","))
    cfg = Config(brands=tuple(args.brands.split(",")), types=types,
                 f=tuple([1 / len(types)] * len(types)), a_max=args.a_max, nocar=args.nocar)
    th = theta(cfg)
    z, ok, _ = solve(th, cfg)
    assert ok, "el equilibrio de Gillingham no converge"
    objs = equilibrium_objects(z, th, cfg)
    J, A, n_act, n = dims(cfg)
    qa = sum(f * split_states(o["q"], cfg)[0] for o, f in zip(objs, cfg.f))
    kq = sum(f * split_states(o["q"], cfg)[0] * split_states(o["keep"], cfg)[0] for o, f in zip(objs, cfg.f))
    jj, aa = np.meshgrid(np.arange(J), np.arange(1, A), indexing="ij")
    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame(dict(j=jj.ravel(), a=aa.ravel(), q=qa.ravel(), P=objs[0]["P"].ravel(),
                      keep=(kq / qa).ravel())).to_csv(os.path.join(args.outdir, "gill_por_edad.csv"), index=False)
    with open(os.path.join(args.outdir, "gill_resumen.json"), "w", encoding="utf-8") as fh:
        json.dump(market_stats(objs, th, cfg), fh, indent=1)
    print(f"Gillingham exportado a {args.outdir}")


if __name__ == "__main__":
    main()
