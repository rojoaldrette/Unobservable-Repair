# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/comparacion/compare.py
# Goal:           Comparar el equilibrio de la réplica de Gillingham con el de modelo_fin
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
Uso (desde claude/scripts/comparacion/):
    PYTHONIOENCODING=utf-8 python compare.py [--n_s 24] [--skip-mine]

Corre estos equilibrios y escribe output/comparacion.csv (una fila por modelo x edad):
    G25      Gillingham, a_max = 25 (el del paper)
    M25      modelo_fin con la calibración "tesis" (calibracion.py), un tipo de hogar
    M25_sin_chatarreo  lo mismo con scrap = False
  con --viejos, además los de v1.1:
    G7       Gillingham, a_max = 7
    M7       modelo_fin con sus defaults (salvo n_s)
    M7_gill  modelo_fin con s calibrada a la logit de accidentes de Gillingham (Tabla 4)
             y Ts con la lectura raw de la Tabla 5 (0.9106 / 2.1929)

Los dos paquetes usan los mismos nombres de módulo (params, bellman, ...), así que se
cargan uno a la vez y se limpian de sys.modules entre cargas.
'''

import argparse
import dataclasses
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Los dos paquetes comparten estos nombres de módulo: se limpian de sys.modules entre cargas
MODULES = ("params", "utils", "primitives", "bellman", "probabilities", "transitions", "ED",
           "theta", "equilibrium", "calibracion")


def load(pkg):
    for name in MODULES:
        sys.modules.pop(name, None)
    path = os.path.join(HERE, "..", pkg)
    sys.path.insert(0, path)
    try:
        mods = {}
        for name in MODULES:
            if os.path.exists(os.path.join(path, name + ".py")):
                mods[name] = importlib.import_module(name)
        return mods
    finally:
        sys.path.remove(path)


def run_gillingham(a_max):
    m = load("gillingham")
    g = dataclasses.replace(m["params"].GParams(), a_max=a_max)
    eq = m["ED"].solve_equilibrium(g)
    mk = m["ED"].market_components(eq.EV, eq.P, g)
    qa, qt, qn = m["utils"].split_states(mk.q, g)
    qa = np.asarray(qa)
    rows = []
    for j in range(g.n_brands):
        for a in range(a_max - 1):
            rows.append(dict(j=j, a=a + 1, P=float(eq.P[j, a]), q=float(qa[j, a]),
                             keep=float(mk.keep[j, a]), trade=float(mk.trade[j, a]),
                             purge=float(mk.purge[j, a]), endo_scrap=float(mk.endo_scrap[j, a]),
                             accident=float(mk.accident[j, a + 1]), repair=np.nan))
    agg = dict(no_car=float(qn), trade_mass=float(mk.trade_mass),
               new_share=float(np.sum(mk.buy_new)), converged=eq.converged)
    return pd.DataFrame(rows), agg


def gill_like_s(g):
    # Media de s por edad parecida a logit(-5.62 + 0.18 a) de Gillingham en edades bajas
    return dataclasses.replace(g, s_max=0.1, s_new=0.004, s_const=(0.002, 0.002, 0.002),
                               s_age=0.0003, s_sigma=0.003, s_repair=0.004,
                               tc_sell=0.9106, tc_sell_inspect=2.1929)


def run_mine(n_s, gill_s=False):
    m = load("modelo_fin")
    g = dataclasses.replace(m["params"].Params(), n_s=n_s)
    if gill_s:
        g = gill_like_s(g)
    eq = m["ED"].solve_equilibrium(g)
    mk = m["ED"].market_components(eq.EV, eq.P, g)
    qa, qt, qn = m["utils"].split_states(mk.q, g)
    qa, P = np.asarray(qa), np.asarray(eq.P)
    grid = np.asarray(m["utils"].make_s_grid(g))
    rows = []
    for j in range(g.n_brands):
        for a in range(g.a_max - 1):
            w = qa[j, a] / qa[j, a].sum()
            avg = lambda x: float(np.sum(w * np.asarray(x)[j, a]))
            rows.append(dict(j=j, a=a + 1, P=float(np.sum(w * P[j, a])), q=float(qa[j, a].sum()),
                             keep=avg(mk.keep), trade=avg(mk.trade), purge=avg(mk.purge),
                             endo_scrap=0.0, accident=float(np.sum(w * grid)),
                             repair=avg(mk.repair)))
    agg = dict(no_car=float(qn), trade_mass=float(mk.trade_mass),
               new_share=float(np.sum(mk.buy_new)), converged=eq.converged)
    return pd.DataFrame(rows), agg


def run_tesis(n_s, a_max=25, scrap=True):
    # modelo_fin con la calibración "tesis" (log-odds, parámetros de Gillingham, chatarreo),
    # un tipo de hogar (el de G25), equilibrio con equilibrium.solve
    m = load("modelo_fin")
    g = dataclasses.replace(m["calibracion"].calibracion("tesis", a_max=a_max, n_s=n_s), scrap=scrap)
    eco = m["theta"].economy(g, m["params"].Types(("low_couple_poor",), (1.0,)))
    z, ok, _ = m["equilibrium"].solve(eco, method="krylov")
    o = m["equilibrium"].equilibrium_objects(z, eco)[0]
    sp = lambda k: np.asarray(m["utils"].split_states(o[k], g)[0])
    qa, P = sp("q"), np.asarray(o["P"])
    grid = np.asarray(m["utils"].make_s_grid(g))
    keep, trade, purge, scrap_, rep = sp("keep"), sp("trade"), sp("purge"), sp("scrap"), sp("repair")
    rows = []
    for j in range(g.n_brands):
        for a in range(a_max - 1):
            w = qa[j, a] / max(qa[j, a].sum(), 1e-300)
            avg = lambda x: float(np.sum(w * x[j, a]))
            rows.append(dict(j=j, a=a + 1, P=avg(P), q=float(qa[j, a].sum()),
                             keep=avg(keep), trade=avg(trade), purge=avg(purge),
                             endo_scrap=avg((1 - keep) * scrap_), accident=float(np.sum(w * grid)),
                             repair=avg(rep)))
    q = np.asarray(o["q"])
    agg = dict(no_car=float(q[-1]), trade_mass=float(q @ np.asarray(o["trade"])),
               new_share=float(np.sum(np.asarray(o["buy"])[-g.n_brands - 1:-1])), converged=ok)
    return pd.DataFrame(rows), agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_s", type=int, default=40, help="grid de s de modelo_fin")
    ap.add_argument("--viejos", action="store_true", help="también G7, M7 y M7_gill (v1.1)")
    ap.add_argument("--skip-mine", action="store_true")
    args = ap.parse_args()

    # Default: Gillingham con a_max = 25 contra modelo_fin con la calibración "tesis"
    # (con chatarreo, y sin chatarreo para ver su efecto)
    runs = [("G25", lambda: run_gillingham(25))]
    if not args.skip_mine:
        runs += [("M25", lambda: run_tesis(args.n_s)),
                 ("M25_sin_chatarreo", lambda: run_tesis(args.n_s, scrap=False))]
    if args.viejos:
        runs += [("G7", lambda: run_gillingham(7))]
        if not args.skip_mine:
            runs += [("M7", lambda: run_mine(args.n_s)), ("M7_gill", lambda: run_mine(args.n_s, True))]

    tables, summary = [], []
    for name, fn in runs:
        t0 = time.time()
        df, agg = fn()
        df.insert(0, "modelo", name)
        tables.append(df)
        summary.append(dict(modelo=name, **agg, segundos=round(time.time() - t0)))
        print(f"{name}: {summary[-1]}", flush=True)

    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    full = pd.concat(tables, ignore_index=True)
    full.to_csv(os.path.join(HERE, "output", "comparacion.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(HERE, "output", "comparacion_resumen.csv"), index=False)

    pd.set_option("display.width", 200)
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nj = 0 (light brown), edades 1, 3, 6, 10, 15, 20, 24:")
    cols = ["modelo", "a", "P", "q", "keep", "endo_scrap", "accident", "repair"]
    sel = (full["j"] == 0) & full["a"].isin([1, 3, 6, 10, 15, 20, 24])
    print(full[sel][cols].round(4).to_string(index=False))
