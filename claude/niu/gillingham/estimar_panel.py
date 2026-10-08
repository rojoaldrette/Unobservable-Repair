# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/gillingham/estimar_panel.py
# Goal:           Estimar Gillingham sobre un panel externo (el de modelo_tesis)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
    python -u estimar_panel.py --panel <panel.csv.gz> --outdir <carpeta> --rep k
           [--a_max 25 --brands ... --types ... --nocar ... --infos parcial,completa]

El panel trae id, tipo, k, x, o, h en el espacio de estados de Gillingham (sin w ni
reparación; lo arma modelo_tesis/montecarlo.py).  Se estima desde los parámetros del paper
(los mismos que son la verdad en modelo_tesis), con las dos verosimilitudes:

    "parcial"   (la del paper): no distingue accidente de chatarreo voluntario
    "completa": sí lo distingue

La diferencia entre la estimación y esos parámetros es el sesgo de ignorar desgaste y
reparación.  Agrega filas a <carpeta>/gill_parametros_reps*.csv y gill_resumen_reps*.csv
(el sufijo lo da --bloque).  Corre en CPU (lo lanza montecarlo.py con JAX_PLATFORMS=cpu).
'''

import argparse
import os

import numpy as np
import pandas as pd

from params import Config, theta
from equilibrio import solve, equilibrium_objects, market_stats
from gen_dataset import to_cells
from ll_estim import free_spec, pack, natural, labels, cell_data, LLEval, estimate, unpack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--bloque", default="")
    ap.add_argument("--a_max", type=int, default=Config.a_max)
    ap.add_argument("--brands", default=",".join(Config.brands))
    ap.add_argument("--types", default=",".join(Config.types))
    ap.add_argument("--nocar", default=Config.nocar)
    ap.add_argument("--infos", default="parcial,completa")
    ap.add_argument("--lbfgs_iter", type=int, default=500)
    ap.add_argument("--bhhh_iter", type=int, default=50)
    args = ap.parse_args()
    types = tuple(args.types.split(","))
    cfg = Config(brands=tuple(args.brands.split(",")), types=types,
                 f=tuple([1 / len(types)] * len(types)), a_max=args.a_max, nocar=args.nocar)
    th0 = theta(cfg)
    spec = free_spec(th0)
    x0 = pack(th0, spec, cfg)
    truth = np.asarray(natural(x0, spec, th0, cfg))
    z0, ok, _ = solve(th0, cfg)
    df = pd.read_csv(args.panel)

    par, res = [], []
    for info in args.infos.split(","):
        cells = to_cells(df, info)
        ev = LLEval(cfg, spec, th0, cell_data(cells, info), info, z0=z0)
        r = estimate(ev, x0, args.lbfgs_iter, args.bhhh_iter)
        for lab, tv, e, s in zip(labels(spec), truth, r["theta"], r["se"]):
            par.append(dict(rep=args.rep, estimador=f"gill_{info}", arranque=0, mejor=True,
                            parametro=lab, verdad=tv, estimado=e, se=s, ll=r["ll"],
                            convergio=r["converged"]))
        th_hat = unpack(np.asarray(r["x"]), spec, th0, cfg)
        st = market_stats(equilibrium_objects(r["z"], th_hat, cfg), th_hat, cfg)
        res.append(dict(rep=args.rep, estimador=f"gill_{info}", ll=r["ll"],
                        n_obs=int(cells["cnt"].sum()), n_celdas=len(cells), convergio=r["converged"],
                        n_eval=r["n_eval"], segundos=r["seconds"], cond_B=r["cond_B"],
                        **{f"{k}_est": v for k, v in st.items()}))
        print(f"  Gillingham [{info}]: LL = {r['ll']:.2f}, convergió = {r['converged']}, "
              f"{r['seconds']:.0f} s", flush=True)
    for name, d in (("gill_parametros", par), ("gill_resumen", res)):
        path = os.path.join(args.outdir, f"{name}{args.bloque}.csv")
        pd.DataFrame(d).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


if __name__ == "__main__":
    main()
