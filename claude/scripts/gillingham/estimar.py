# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/estimar.py
# Goal:           Estimación con la verosimilitud de Gillingham y salidas en CSV
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           06/10/2026
#
# _____________________________________________________________________________
'''
Uso (desde claude/scripts/gillingham/):

  # Datos de Gillingham (su propio modelo), réplica 0
  python -u estimar.py --reps 0:1 -v

  # Datos de modelo_fin (con reparación): el ejercicio del sesgo.  El panel sale de
  # modelo_fin/estimar.py --guardar_panel.  a_max y tipos tienen que coincidir.
  python -u estimar.py --panel ../../output/estimaciones/modelo_fin/<tag>/panel_rep0.csv.gz --a_max 7 -v

  # Prueba de humo
  python -u estimar.py --smoke

Estimadores (los de loglikelihood.py):
  gill_parcial   la verosimilitud del paper (apéndice D): sin precios ni accidentes
  gill_completa  oráculo: ve el estado, accidentes incluidos

Con --panel los datos vienen de modelo_fin: s y r se borran, los estados se pasan al
layout (j, a) de Gillingham y los años-régimen se juntan (Gillingham no ve R).  La
"verdad" de los parámetros son los de PAPER_TYPES / GParams, que son los mismos que usa
modelo_fin para utilidades y costos; los accidentes y sigma_sell no tienen contraparte
exacta en modelo_fin (allá la prob. de accidente es s y no hay chatarreo endógeno).

Salidas en claude/output/estimaciones/gillingham/<tag>/ (formato largo, como modelo_fin):
  parametros_reps<a>-<b>.csv   rep, estimador, parametro, verdad, estimado, se, z, z_vs_verdad, ic95_lo, ic95_hi, ll, convergio
  resumen_reps<a>-<b>.csv      rep, estimador, ll, n_obs, n_celdas, convergio, iters, segundos, mismo_optimo,
                   sin_coche, tasa de chatarreo endógeno, accidentes, P_rmse (solo con datos propios)
  precios.csv      fuente, j, a, P, q          fuente ∈ {verdad, gill_parcial, gill_completa}
  ccps.csv         fuente, tipo, j, a, keep, purge, trade, scrap, q
  distribucion.csv fuente, tipo, estado, j, a, q
  config.json
'''

import argparse
import dataclasses
import json
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import pandas as pd
import jax.numpy as jnp

from params import GParams, GTypes
from utils import dims, split_states
from theta import FIELDS, economy, theta_types, free_spec, pack, natural, labels
from simulate import simulate_panel, to_cells
from montecarlo import MCDesign, prepare, estimate, market_stats

OUTDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "..", "output", "estimaciones", "gillingham"))
INFO_NAME = {"parcial": "gill_parcial", "completa": "gill_completa"}


# Datos de modelo_fin -> layout de Gillingham ______________________________________________________

def cells_from_modelo_fin(df, g):
    # Panel de modelo_fin (gen_dataset.simulate_economy) -> celdas de Gillingham.
    # X: act(j, a) = j (A-1) + a - 1,  term(j) = n_act + j,  none = n - 1.  H igual con new(j).
    # o: keep 0; purge 1 (+1 si se deshace de un terminal: chatarra); trade 3 (+1 igual).
    J, A, n_act, n = dims(g)
    est = df["estado"].to_numpy()
    x = np.where(est == "activo", df["j"] * (A - 1) + df["a"] - 1,
                 np.where(est == "terminal", n_act + df["j"], n - 1))
    th = df["tipo_h"].to_numpy()
    h = np.where(th == "usado", df["j_h"] * (A - 1) + df["d_h"] - 1,
                 np.where(th == "nuevo", n_act + df["j_h"], n - 1))
    # chatarra: deshacerse de un terminal, o chatarreo endógeno de un activo (si modelo_fin lo tiene)
    chat = df["chatarreo"].to_numpy() == 1 if "chatarreo" in df else np.zeros(len(df), bool)
    term = ((est == "terminal") | chat).astype(int)
    dec = df["decision"].to_numpy()
    o = np.where(dec == "keep", 0, np.where(dec == "trade", 3 + term, 1 + term))
    d = pd.DataFrame(dict(id_hogar=df["id_hogar"].to_numpy(), tipo=df["tipo"].to_numpy(),
                          k=df["k"].to_numpy(), x=x.astype(np.int64), o=o, h=h.astype(np.int64)))
    d = d.sort_values(["id_hogar", "k"], kind="stable", ignore_index=True)
    return to_cells(d)


# Salidas ______________________________________________________________

def tablas_equilibrio(fuente, e, eco):
    # e: segundo valor de market_stats (P, qa, eqos)
    g = eco.g
    J, A, n_act, n = dims(g)
    jj, aa = np.meshgrid(np.arange(J), np.arange(1, A), indexing="ij")
    precios = pd.DataFrame(dict(fuente=fuente, j=jj.ravel(), a=aa.ravel(),
                                P=np.ravel(e["P"]), q=np.ravel(e["qa"])))
    ccp, dist = [], []
    for tau, o in enumerate(e["eqos"]):
        qa, qt, qn = (np.asarray(v) for v in split_states(jnp.asarray(o["q"]), g))
        cols = {k: np.asarray(split_states(jnp.asarray(o[k]), g)[0]).ravel()
                for k in ("keep", "purge", "trade", "scrap_x")}
        cols["scrap"] = cols.pop("scrap_x")
        ccp.append(pd.DataFrame(dict(fuente=fuente, tipo=tau, j=jj.ravel(), a=aa.ravel(), **cols,
                                     q=qa.ravel())))
        dist.append(pd.DataFrame(dict(fuente=fuente, tipo=tau, estado="activo", j=jj.ravel(),
                                      a=aa.ravel(), q=qa.ravel())))
        dist.append(pd.DataFrame(dict(fuente=fuente, tipo=tau, estado="terminal", j=np.arange(J),
                                      a=A, q=qt)))
        dist.append(pd.DataFrame(dict(fuente=fuente, tipo=tau, estado="sin_coche", j=[-1], a=[-1],
                                      q=[float(qn)])))
    return precios, pd.concat(ccp), pd.concat(dist)


def _append(df, path):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# Main ______________________________________________________________

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0:1")
    ap.add_argument("--panel", default="", help="panel de modelo_fin (csv.gz); si no, datos propios")
    ap.add_argument("--N", type=int, default=20_000)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--a_max", type=int, default=25)          # el del paper
    ap.add_argument("--infos", default="parcial,completa")
    ap.add_argument("--fix", default="")
    ap.add_argument("--types", default="low_couple_poor,low_single_poor")
    ap.add_argument("--f", default="")
    ap.add_argument("--n_starts", type=int, default=2)
    ap.add_argument("--perturb", type=float, default=0.1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse()
    if args.smoke:
        args.N, args.K, args.a_max, args.reps, args.n_starts = 3_000, 4, 7, "0:1", 0
        args.verbose, args.tag = True, args.tag or "smoke"
    g = dataclasses.replace(GParams(), a_max=args.a_max)
    names = tuple(args.types.split(","))
    f = tuple(map(float, args.f.split(","))) if args.f else tuple([1.0 / len(names)] * len(names))
    fix = set(filter(None, args.fix.split(",")))
    design = MCDesign(N=args.N, K=args.K, infos=tuple(args.infos.split(",")),
                      free=tuple(k for k in FIELDS if k not in fix), n_starts=args.n_starts,
                      perturb=args.perturb, types=GTypes(names, f))
    datos = "modelo_fin" if args.panel else "gillingham"
    tag = args.tag or f"datos_{datos}_{design.tag(g)}"
    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(args), datos=datos, params=dataclasses.asdict(g),
                       tipos=dict(names=names, f=f)), fh, indent=1, default=str)

    z_true, e0, stats0 = prepare(g, design, outdir, verbose=args.verbose)
    th0 = theta_types(g, design.types)
    spec = free_spec(th0, design.free)
    truth = np.asarray(natural(jnp.asarray(pack(th0, spec)), spec, th0))
    labs = labels(spec)
    panel = pd.read_csv(args.panel) if args.panel else None

    a, b = map(int, args.reps.split(":"))
    for rep in range(a, b):
        rng = np.random.default_rng(10_000 + rep)
        if panel is not None:
            cells = cells_from_modelo_fin(panel, g)
        else:
            cells = to_cells(simulate_panel(e0["eqos"], g, f, args.N, args.K, seed=rep))
        par_rows, res_rows, tabs = [], [], []
        if rep == a:
            tabs.append(tablas_equilibrio("verdad", e0, economy(g, design.types)))
        for info in design.infos:
            name = INFO_NAME[info]
            r = estimate(g, design, cells, info, z_true, rng, verbose=args.verbose)
            for lab, tv, e, s in zip(labs, truth, r["theta"], r["se"]):
                ok = np.isfinite(s) and s > 0
                par_rows.append(dict(rep=rep, estimador=name, parametro=lab, verdad=tv, estimado=e,
                                     se=s, z=e / s if ok else np.nan,
                                     z_vs_verdad=(e - tv) / s if ok else np.nan,
                                     ic95_lo=e - 1.96 * s, ic95_hi=e + 1.96 * s,
                                     ll=r["ll"], convergio=r["converged"]))
            stats, e = market_stats(r["z"], r["eco"])
            row = dict(rep=rep, estimador=name, datos=datos, ll=r["ll"], n_obs=int(cells["cnt"].sum()),
                       n_celdas=len(cells), convergio=r["converged"], iters=r["iters"],
                       segundos=r["seconds"], mismo_optimo=f"{r['n_same']}/{r['n_fits']}")
            row.update({f"{k}_est": v for k, v in stats.items()})
            row.update({f"{k}_verdad_gill": v for k, v in stats0.items()})
            if panel is None:
                dP = np.ravel(e["P"]) - np.ravel(e0["P"])
                qa = np.ravel(e0["qa"])
                row["P_rmse"] = float(np.sqrt(np.sum(qa * dP ** 2) / np.sum(qa)))
            res_rows.append(row)
            if rep == a:
                tabs.append(tablas_equilibrio(name, e, r["eco"]))
            if args.verbose:
                print(f"rep {rep} [{name}]: LL = {r['ll']:.2f}")
        sufijo = f"reps{a}-{b - 1}" if panel is None else "reps0-0"
        _append(pd.DataFrame(par_rows), os.path.join(outdir, f"parametros_{sufijo}.csv"))
        _append(pd.DataFrame(res_rows), os.path.join(outdir, f"resumen_{sufijo}.csv"))
        if tabs:
            for i, nm in enumerate(("precios", "ccps", "distribucion")):
                pd.concat([t[i] for t in tabs]).to_csv(os.path.join(outdir, f"{nm}.csv"), index=False)
        if panel is not None:
            break                       # un solo panel: una sola estimación
    print(f"listo: {outdir}")


if __name__ == "__main__":
    main()
