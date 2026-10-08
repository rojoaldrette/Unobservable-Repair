# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/gillingham/por_edad_mc.py
# Goal:           Equilibrio de Gillingham en cada θ estimado del MC, por (marca, edad)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
    python -u por_edad_mc.py --mc <carpeta del MC> --reps 0:50 [--a_max ... --brands ... --types ... --nocar ...]

Lo lanza modelo_tesis/por_edad_mc.py (proceso aparte, CPU: los dos paquetes comparten
nombres de módulo).  Lee <mc>/gill_parametros_reps*.csv y agrega filas a
<mc>/gill_por_edad_mc.csv: rep, estimador, j, a, q, P, keep (como exportar.py; q y keep
agregados sobre tipos con pesos f).  rep = -1, estimador = gill_verdad: Gillingham en los
parámetros verdaderos.  Lo que ya está se salta.
'''

import argparse
import glob
import os

import numpy as np
import pandas as pd
import jax.numpy as jnp

from params import Config, theta
from utils import dims, split_states
from equilibrio import solve, solve_or_restart, equilibrium_objects
from ll_estim import free_spec, labels


def por_edad(z, th, cfg, rep, estimador):
    objs = equilibrium_objects(z, th, cfg)
    J, A, n_act, n = dims(cfg)
    qa = sum(f * split_states(o["q"], cfg)[0] for o, f in zip(objs, cfg.f))
    kq = sum(f * split_states(o["q"], cfg)[0] * split_states(o["keep"], cfg)[0] for o, f in zip(objs, cfg.f))
    jj, aa = np.meshgrid(np.arange(J), np.arange(1, A), indexing="ij")
    return pd.DataFrame(dict(rep=rep, estimador=estimador, j=jj.ravel(), a=aa.ravel(), q=qa.ravel(),
                             P=objs[0]["P"].ravel(), keep=(kq / qa).ravel()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", required=True)
    ap.add_argument("--reps", default="0:50")
    ap.add_argument("--a_max", type=int, default=Config.a_max)
    ap.add_argument("--brands", default=",".join(Config.brands))
    ap.add_argument("--types", default=",".join(Config.types))
    ap.add_argument("--nocar", default=Config.nocar)
    args = ap.parse_args()
    types = tuple(args.types.split(","))
    cfg = Config(brands=tuple(args.brands.split(",")), types=types,
                 f=tuple([1 / len(types)] * len(types)), a_max=args.a_max, nocar=args.nocar)
    a, b = map(int, args.reps.split(":"))
    path = os.path.join(args.mc, "gill_por_edad_mc.csv")
    hechos = set()
    if os.path.exists(path):
        h = pd.read_csv(path, usecols=["rep", "estimador"])
        hechos = set(zip(h.rep, h.estimador))

    th0 = theta(cfg)
    spec = free_spec(th0)
    labs = labels(spec)
    z0, ok, _ = solve(th0, cfg)
    assert ok, "el equilibrio de Gillingham no converge en la verdad"
    if (-1, "gill_verdad") not in hechos:
        por_edad(z0, th0, cfg, -1, "gill_verdad").to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    files = sorted(glob.glob(os.path.join(args.mc, "gill_parametros_reps*.csv")))
    p = pd.concat(map(pd.read_csv, files), ignore_index=True)
    p = p[p["mejor"] & (p.rep >= a) & (p.rep < b)].drop_duplicates(["rep", "estimador", "parametro"], keep="last")
    for (rep, est), d in p.groupby(["rep", "estimador"]):
        if (rep, est) in hechos:
            continue
        v = d.set_index("parametro").loc[labs, "estimado"].to_numpy()
        th, i = dict(th0), 0
        for k, shape in spec:
            size = int(np.prod(shape))
            th[k] = jnp.asarray(v[i:i + size]).reshape(shape)
            i += size
        z, ok, _ = solve_or_restart(th, cfg, z0)
        if not ok:
            print(f"  OJO: Gillingham rep {rep} [{est}] no converge; se omite", flush=True)
            continue
        por_edad(z, th, cfg, int(rep), est).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print(f"Gillingham por edad listo: {path}", flush=True)


if __name__ == "__main__":
    main()
