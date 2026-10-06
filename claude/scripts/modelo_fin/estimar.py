# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/estimar.py
# Goal:           Estimación estructural de modelo_fin (D0 y D1) y salidas en CSV
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           06/10/2026
#
# _____________________________________________________________________________
'''
Uso (desde claude/scripts/modelo_fin/):

  # Una estimación (réplica 0) con los dos estimadores; guarda el panel para Gillingham
  python -u estimar.py --reps 0:1 --guardar_panel -v

  # Monte Carlo por bloques (p. ej. un array de SLURM)
  python -u estimar.py --reps 0:10 --N 20000 --K 2 --T 13
  python -u estimar.py --reps 10:20 ...

  # Prueba de humo (tamaño de juguete, segundos a minutos)
  python -u estimar.py --smoke

Flujo:
1. `calibracion`: Params de la corrida (ver abajo).  `economy`: dos tipos de hogar.
2. `verdad`: equilibrio verdadero de cada régimen t (R_t = R · exp(zeta_t)), con
   arranque en caliente de un régimen al siguiente.  Se guarda (y se reutiliza) en
   verdad.npz.
3. Por réplica: simula el panel (gen_dataset.simulate_economy), arma las celdas y estima
   con "oraculo" (D0: r observada) y "hx" (D1: r no observada).  Arranques: la verdad
   (con el equilibrio verdadero) y n_starts perturbados; se queda el de mayor LL y se
   guardan TODOS los arranques.
4. Escribe CSV en claude/output/estimaciones/modelo_fin/<tag>/ (ver `escribir_*`).

Salidas (todas en formato largo, para pandas):
  parametros_reps<a>-<b>.csv   (un archivo por bloque de réplicas)
                     rep, estimador, arranque, mejor, parametro, verdad, estimado, se, z, p, ic95_lo, ic95_hi
  resumen_reps<a>-<b>.csv      rep, estimador, ll, n_obs, n_celdas, convergio, iters, segundos, arranques,
                     mismo_optimo, y estadísticas de mercado verdaderas y estimadas
  precios.csv        (réplica rep_reporte) fuente, regimen, zeta, j, a, s_idx, s, P, q, sin_masa
  distribucion.csv   (réplica rep_reporte, régimen central) fuente, regimen, tipo, estado, j, a, s_idx, s, q
  ccps.csv           (réplica rep_reporte, régimen central) fuente, regimen, tipo, j, a, s_idx, s,
                     keep, purge, trade, repair, q
  config.json        argumentos, Params y tipos
  panel_rep<k>.csv.gz  (con --guardar_panel) el panel simulado completo, para
                     gillingham/estimar.py --panel
fuente ∈ {verdad, oraculo, hx}.

Calibración (`--calib`, provisional hasta la Fase 3: log-odds, a_max = 25):
  defaults  Params() tal cual (s en niveles, accidentes altos)
  gill_s    s calibrada a la logit de accidentes de Gillingham en edades bajas
            (como comparacion/compare.py, M7_gill)
  tesis_v0  gill_s + R(j, a) realista (4 / 6.5 / 10 mil DKK en a = 1, +6% por año)
            y sigma_repair = 0.3 (por calibrar: que Pr(repair) tenga información)
'''

import argparse
import dataclasses
import json
import os
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import pandas as pd
import jax.numpy as jnp

from params import Params, Types
from utils import dims, split_states, make_s_grid, s_new_index
from primitives import s_transition
from theta import FIELDS, economy, free_spec, pack, natural, labels, Economy, unpack
from equilibrium import solve, equilibrium_objects, split_z
from gen_dataset import simulate_economy
from estructural import INFOS, treat_data, LLEval, estimate

OUTDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "..", "output", "estimaciones", "modelo_fin"))


# Calibración ______________________________________________________________

def repair_prices(a_max, base, growth):
    # R(j, a) = base_j (1 + growth (a − 1)),  a = 1..a_max−1
    return tuple(tuple(round(b * (1 + growth * (a - 1)), 4) for a in range(1, a_max)) for b in base)


def calibracion(nombre, a_max=7, n_s=100):
    g = dataclasses.replace(Params(), a_max=a_max, n_s=n_s,
                            repair_price=repair_prices(a_max, (15.0, 12.5, 25.0), 0.15))
    if nombre == "defaults":
        return g
    g = dataclasses.replace(g, s_max=0.1, s_new=0.004, s_const=(0.002, 0.002, 0.002),
                            s_age=0.0003, s_sigma=0.003, s_repair=0.004)
    if nombre == "gill_s":
        return g
    if nombre == "tesis_v0":
        return dataclasses.replace(g, repair_price=repair_prices(a_max, (4.0, 6.5, 10.0), 0.06),
                                   sigma_repair=0.3)
    raise ValueError(nombre)


# Verdad ______________________________________________________________

def regimes_R(g, T, spread):
    zetas = np.linspace(-spread, spread, T) if T > 1 else np.zeros(1)
    R = np.asarray(g.repair_price)
    return zetas, np.stack([R * np.exp(z) for z in zetas])


def verdad(g, types, Rs, path, method, verbose=False):
    if os.path.exists(path):
        with np.load(path) as f:
            if f["Rs"].shape == Rs.shape and np.allclose(f["Rs"], Rs):
                return jnp.asarray(f["zs"])
    eco = economy(g, types)
    zs, z = [], None
    for t, R in enumerate(Rs):
        if verbose:
            print(f"verdad, régimen {t}")
        z, ok, it = solve(eco.with_repair_price(R), z, method=method, verbose=verbose)
        if not ok:
            raise RuntimeError(f"el equilibrio verdadero del régimen {t} no converge")
        zs.append(z)
    zs = jnp.stack(zs)
    np.savez_compressed(path, zs=np.asarray(zs), Rs=Rs)
    return zs


# Estadísticas de mercado ______________________________________________________________

def mercado(zs, ecos):
    # Promedio sobre regímenes de estadísticas agregadas (ponderadas por f)
    g = ecos[0].g
    J, A, S, n_act, n = dims(g)
    grid = np.asarray(make_s_grid(g))
    s0 = float(grid[s_new_index(g)])
    stats = dict(sin_coche=0.0, tasa_reparacion=0.0, tasa_accidentes=0.0, edad_media=0.0,
                 precio_medio=0.0)
    for z, eco in zip(zs, ecos):
        objs = equilibrium_objects(z, eco)
        P = objs[0]["P"]
        rep = acc = cars = age = qa_tot = pq = none = 0.0
        for o, f in zip(objs, eco.f):
            q = o["q"]
            e_none = np.zeros(n)
            e_none[-1] = 1.0
            q_hold = o["keep"] * q + (q @ o["trade"]) * o["buy"] + (q @ o["purge"]) * e_none
            hu, hn, _ = split_states(q_hold, g)
            hu, hn = np.asarray(hu), np.asarray(hn)
            pr, _, _ = split_states(o["repair"], g)
            rep += f * float(np.sum(hu * np.asarray(pr)))
            # accidentes: usados que no están en la última edad (prob s) y nuevos (prob s_new)
            acc += f * float(np.sum(hu[:, :A - 2] * grid) + np.sum(hn) * s0)
            cars += f * float(hu.sum() + hn.sum())
            qa, _, qn = split_states(q, g)
            qa = np.asarray(qa)
            age += f * float(np.sum(qa.sum(axis=(0, 2)) * np.arange(1, A)))
            qa_tot += f * float(qa.sum())
            pq += f * float(np.sum(qa * np.asarray(P)))
            none += f * float(qn)
        R_ = len(zs)
        stats["sin_coche"] += none / R_
        stats["tasa_reparacion"] += rep / cars / R_
        stats["tasa_accidentes"] += acc / cars / R_
        stats["edad_media"] += age / qa_tot / R_
        stats["precio_medio"] += pq / qa_tot / R_
    return stats


def precio_rmse(zs_hat, zs_true, ecos):
    # RMSE de P(θ̂) contra P verdadero, ponderado por q verdadero (agregado sobre tipos)
    num = den = 0.0
    for zh, zt, eco in zip(zs_hat, zs_true, ecos):
        _, Ph = split_z(zh, eco)
        _, Pt = split_z(zt, eco)
        w = sum(f * np.asarray(split_states(o["q"], eco.g)[0])
                for o, f in zip(equilibrium_objects(zt, eco), eco.f))
        num += float(np.sum(w * (np.asarray(Ph) - np.asarray(Pt)) ** 2))
        den += float(np.sum(w))
    return float(np.sqrt(num / den))


# Salidas de objetos de equilibrio ______________________________________________________________

def tablas_equilibrio(fuente, zs, ecos, zetas, regimen_central):
    # precios (todos los regímenes), distribución y CCPs (régimen central)
    g = ecos[0].g
    J, A, S, n_act, n = dims(g)
    grid = np.asarray(make_s_grid(g))
    jj, aa, ss = np.meshgrid(np.arange(J), np.arange(1, A), np.arange(S), indexing="ij")
    base = dict(j=jj.ravel(), a=aa.ravel(), s_idx=ss.ravel(), s=grid[ss.ravel()])
    precios, dist, ccp = [], [], []
    for t, (z, eco) in enumerate(zip(zs, ecos)):
        objs = equilibrium_objects(z, eco)
        P = np.asarray(objs[0]["P"])
        q_agg = sum(f * np.asarray(split_states(o["q"], g)[0]) for o, f in zip(objs, eco.f))
        oferta = sum(f * np.asarray(split_states(o["q"], g)[0]) * (1 - np.asarray(split_states(o["keep"], g)[0]))
                     for o, f in zip(objs, eco.f))
        precios.append(pd.DataFrame(dict(fuente=fuente, regimen=t, zeta=zetas[t], **base,
                                         P=P.ravel(), q=q_agg.ravel(),
                                         sin_masa=(oferta.ravel() < g.ed_mass_tol))))
        if t != regimen_central:
            continue
        for tau, o in enumerate(objs):
            qa, qt, qn = split_states(o["q"], g)
            dist.append(pd.DataFrame(dict(fuente=fuente, regimen=t, tipo=tau, estado="activo",
                                          **base, q=np.asarray(qa).ravel())))
            dist.append(pd.DataFrame(dict(fuente=fuente, regimen=t, tipo=tau, estado="terminal",
                                          j=np.arange(J), a=A, s_idx=-1, s=np.nan, q=np.asarray(qt))))
            dist.append(pd.DataFrame(dict(fuente=fuente, regimen=t, tipo=tau, estado="sin_coche",
                                          j=[-1], a=[-1], s_idx=[-1], s=[np.nan], q=[float(qn)])))
            cols = {k: np.asarray(split_states(o[k], g)[0]).ravel() for k in ("keep", "purge", "trade", "repair")}
            ccp.append(pd.DataFrame(dict(fuente=fuente, regimen=t, tipo=tau, **base, **cols,
                                         q=np.asarray(qa).ravel())))
    return pd.concat(precios), pd.concat(dist), pd.concat(ccp)


# Una réplica ______________________________________________________________

def estimar_info(info, df, g, types, th0, spec, Rs, zs_true, rng, args):
    data, cells = treat_data(df, g, info)
    x0 = np.asarray(pack(th0, spec))
    starts = [x0] + [x0 + args.perturb * np.abs(x0) * rng.standard_normal(len(x0))
                     for _ in range(args.n_starts)]
    fits = []
    for s, xs in enumerate(starts):
        ev = LLEval(g, types.f, spec, th0, Rs, data, info,
                    zs0=zs_true if s == 0 else None, method=args.method)
        try:
            r = estimate(ev, xs, args.lbfgs_iter, args.bhhh_iter, verbose=args.verbose)
            r["arranque"] = s
            fits.append(r)
        except (RuntimeError, np.linalg.LinAlgError) as err:
            if args.verbose:
                print(f"  [{info}] arranque {s} falló: {err}")
    if not fits:
        raise RuntimeError(f"[{info}] ningún arranque convergió")
    best = max(fits, key=lambda r: r["ll"])
    n_same = sum(abs(r["ll"] - best["ll"]) < max(1e-4, 1e-8 * abs(best["ll"])) for r in fits)
    return fits, best, n_same, data, cells


def una_replica(rep, g, types, eco0, th0, spec, zetas, Rs, zs_true, F, args, outdir):
    t0 = time.time()
    ecos_true = [eco0.with_repair_price(R) for R in Rs]
    objs = [equilibrium_objects(z, e) for z, e in zip(zs_true, ecos_true)]
    df = simulate_economy(objs, F, g, types.f, args.N, args.K, seed=rep)
    if args.guardar_panel:
        df.to_csv(os.path.join(outdir, f"panel_rep{rep}.csv.gz"), index=False)

    truth = np.asarray(natural(jnp.asarray(pack(th0, spec)), spec, th0))
    labs = labels(spec)
    stats_true = mercado(zs_true, ecos_true)
    rng = np.random.default_rng(10_000 + rep)
    par_rows, res_rows, eq_tabs = [], [], {}
    for info in args.infos:
        fits, best, n_same, data, cells = estimar_info(info, df, g, types, th0, spec, Rs,
                                                       zs_true, rng, args)
        for r in fits:
            for lab, tv, e, s in zip(labs, truth, r["theta"], r["se"]):
                zval = (e - tv) / s if np.isfinite(s) and s > 0 else np.nan
                par_rows.append(dict(rep=rep, estimador=info, arranque=r["arranque"],
                                     mejor=r is best, parametro=lab, verdad=tv, estimado=e, se=s,
                                     z=e / s if np.isfinite(s) and s > 0 else np.nan,
                                     z_vs_verdad=zval, ic95_lo=e - 1.96 * s, ic95_hi=e + 1.96 * s,
                                     ll=r["ll"], convergio=r["converged"]))
        ecos_hat = [Economy(g, dict(unpack(jnp.asarray(best["x"]), spec, th0), repair_price=R), types.f)
                    for R in Rs]
        stats_hat = mercado(best["zs"], ecos_hat)
        row = dict(rep=rep, estimador=info, ll=best["ll"], n_obs=int(cells["cnt"].sum()),
                   n_celdas=len(cells), convergio=best["converged"], iters=best["iters"],
                   segundos=best["seconds"], arranques=len(fits), mismo_optimo=n_same,
                   cond_B=best["cond_B"],
                   P_rmse=precio_rmse(best["zs"], zs_true, ecos_true))
        row.update({f"{k}_verdad": v for k, v in stats_true.items()})
        row.update({f"{k}_est": v for k, v in stats_hat.items()})
        res_rows.append(row)
        eq_tabs[info] = (best["zs"], ecos_hat)
        if args.verbose:
            print(f"rep {rep} [{info}]: LL = {best['ll']:.2f}, P_rmse = {row['P_rmse']:.3f}, "
                  f"{time.time() - t0:.0f} s")
    return pd.DataFrame(par_rows), pd.DataFrame(res_rows), eq_tabs, ecos_true


def _append(df, path):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# Main ______________________________________________________________

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0:1", help="rango a:b de réplicas (semillas)")
    ap.add_argument("--calib", default="tesis_v0", choices=["defaults", "gill_s", "tesis_v0"])
    ap.add_argument("--a_max", type=int, default=7)
    ap.add_argument("--n_s", type=int, default=100)
    ap.add_argument("--types", default="low_couple_poor,low_single_poor")
    ap.add_argument("--f", default="", help="fracciones de cada tipo (default: iguales)")
    ap.add_argument("--T", type=int, default=13, help="regímenes (años) de R")
    ap.add_argument("--spread", type=float, default=0.3, help="R_t = R exp(zeta), zeta en [-spread, spread]")
    ap.add_argument("--N", type=int, default=20_000, help="hogares por régimen")
    ap.add_argument("--K", type=int, default=2, help="años por hogar dentro de un régimen")
    ap.add_argument("--infos", default="oraculo,hx")
    ap.add_argument("--fix", default="", help="parámetros fijos en la verdad, p. ej. s_persist,u_s")
    ap.add_argument("--n_starts", type=int, default=2)
    ap.add_argument("--perturb", type=float, default=0.1)
    ap.add_argument("--method", default="krylov", choices=["krylov", "dense"])
    ap.add_argument("--lbfgs_iter", type=int, default=500)
    ap.add_argument("--bhhh_iter", type=int, default=50)
    ap.add_argument("--rep_reporte", type=int, default=None,
                    help="réplica de la que se guardan precios/distribución/CCPs (default: la primera)")
    ap.add_argument("--guardar_panel", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse()
    if args.smoke:
        # a_max = 7 (con a_max = 4 casi nadie tiene coche y el panel no informa)
        args.a_max, args.n_s, args.T, args.N, args.K = 7, 8, 2, 3_000, 2
        args.n_starts, args.method, args.reps, args.verbose = 1, "dense", "0:1", True
        args.lbfgs_iter, args.bhhh_iter = 15, 3
        args.fix = "u1,u_s,tc_buy_nocar,tc_sell_inspect,s_const,s_age,s_persist"
        args.guardar_panel, args.tag = True, "smoke"
    args.infos = tuple(args.infos.split(","))
    assert all(i in INFOS for i in args.infos), args.infos

    g = calibracion(args.calib, args.a_max, args.n_s)
    names = tuple(args.types.split(","))
    f = tuple(map(float, args.f.split(","))) if args.f else tuple([1.0 / len(names)] * len(names))
    types = Types(names, f)
    eco0 = economy(g, types)
    th0 = eco0.th
    fix = set(filter(None, args.fix.split(",")))
    spec = free_spec(th0, tuple(k for k in FIELDS if k not in fix))
    zetas, Rs = regimes_R(g, args.T, args.spread)

    tag = args.tag or (f"{args.calib}_A{args.a_max}_S{args.n_s}_T{args.T}_N{args.N}_K{args.K}"
                       f"_tipos{len(names)}_p{len(labels(spec))}")
    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(args), params=dataclasses.asdict(g),
                       tipos=dict(names=names, f=f), zetas=zetas.tolist(),
                       libres=labels(spec)), fh, indent=1, default=str)

    zs_true = verdad(g, types, Rs, os.path.join(outdir, "verdad.npz"), args.method, args.verbose)
    F = s_transition(g)
    a, b = map(int, args.reps.split(":"))
    rep_rep = a if args.rep_reporte is None else args.rep_reporte
    central = len(zetas) // 2
    for rep in range(a, b):
        par, res, eq_tabs, ecos_true = una_replica(rep, g, types, eco0, th0, spec, zetas, Rs,
                                                   zs_true, F, args, outdir)
        # Un archivo por bloque de réplicas: los bloques de un array de SLURM no se pisan
        _append(par, os.path.join(outdir, f"parametros_reps{a}-{b - 1}.csv"))
        _append(res, os.path.join(outdir, f"resumen_reps{a}-{b - 1}.csv"))
        if rep == rep_rep:
            tabs = [tablas_equilibrio("verdad", zs_true, ecos_true, zetas, central)]
            tabs += [tablas_equilibrio(info, zs_h, ecos_h, zetas, central)
                     for info, (zs_h, ecos_h) in eq_tabs.items()]
            for i, name in enumerate(("precios", "distribucion", "ccps")):
                pd.concat([t[i] for t in tabs]).to_csv(os.path.join(outdir, f"{name}.csv"), index=False)
    print(f"listo: {outdir}")


if __name__ == "__main__":
    main()
