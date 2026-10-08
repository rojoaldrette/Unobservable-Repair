# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/gillingham/montecarlo.py
# Goal:           Monte Carlo del estimador DNFXP de Gillingham sobre datos simulados
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           07/10/2026
#
# _____________________________________________________________________________
'''
Uso (desde claude/niu/gillingham/):

    python -u montecarlo.py --smoke                    # humo: segundos a pocos minutos
    python -u montecarlo.py --reps 0:100 -v            # MC completo (parámetros del paper)
    python -u montecarlo.py --reps 0:50 --fix tc_buy   # con parámetros fijos
    python -u montecarlo.py --summarize --tag <tag>    # tabla de sesgo, RMSE y cobertura

Por réplica: simula el panel (semilla = rep) y estima con cada `info` ("parcial" = la del
paper, "completa" = oráculo que ve los accidentes).  Arranques: la verdad (con el
equilibrio verdadero) y `n_starts` perturbados (x0 + perturb |x0| N(0, 1)); se queda el de
mayor LL y se guardan todos.

Salidas en claude/niu/output/gillingham/<tag>/ (formato largo, para pandas):
    config.json                  argumentos, Config y θ verdadero
    verdad.json                  estadísticas de mercado del equilibrio verdadero
    parametros_reps<a>-<b>.csv   rep, estimador, arranque, mejor, parametro, verdad, estimado, se, ll, convergio
    resumen_reps<a>-<b>.csv      rep, estimador, ll, n_obs, n_celdas, convergio, n_eval, n_fail,
                                 segundos, arranques, mismo_optimo, cond_B, P_rmse, <estadísticas>_est
    precios.csv                  (primera réplica del bloque) j, a, P_verdad, P_<estimador>
    resumen_mc.csv               (--summarize) sesgo, sd, se mediano, RMSE y cobertura al 95%
'''

import argparse
import dataclasses
import glob
import json
import os
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import pandas as pd
import jax.numpy as jnp

from params import Config
import params
from utils import dims, split_states
from equilibrio import solve, equilibrium_objects, market_stats, split_z
from gen_dataset import simulate_panel, to_cells
from ll_estim import free_spec, pack, natural, labels, cell_data, LLEval, estimate, unpack

OUTDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "output", "gillingham"))
INFOS = ("parcial", "completa")


def precio_rmse(z_hat, z_true, objs_true, cfg):
    # RMSE de P̂ contra P verdadero, ponderado por la masa verdadera de cada (j, a)
    w = sum(f * split_states(o["q"], cfg)[0] for o, f in zip(objs_true, cfg.f))
    d = np.asarray(split_z(z_hat, cfg)[1] - split_z(z_true, cfg)[1])
    return float(np.sqrt(np.sum(w * d ** 2) / np.sum(w)))


def una_replica(rep, cfg, th0, spec, z_true, objs_true, args):
    df = simulate_panel(objs_true, cfg, args.N, args.K, seed=rep)
    truth = np.asarray(natural(pack(th0, spec, cfg), spec, th0, cfg))
    rng = np.random.default_rng(10_000 + rep)
    x0 = np.asarray(pack(th0, spec, cfg))
    starts = [x0] + [x0 + args.perturb * np.abs(x0) * rng.standard_normal(len(x0))
                     for _ in range(args.n_starts)]
    par, res, z_hat = [], [], {}
    for info in args.infos:
        cells = to_cells(df, info)
        data = cell_data(cells, info)
        fits = []
        for s, xs in enumerate(starts):
            if args.verbose:
                print(f"  rep {rep} [{info}] arranque {s}", flush=True)
            ev = LLEval(cfg, spec, th0, data, info, z0=z_true if s == 0 else None)
            try:
                r = estimate(ev, xs, args.lbfgs_iter, args.bhhh_iter, verbose=args.verbose)
            except RuntimeError as err:
                print(f"  rep {rep} [{info}] arranque {s} falló: {err}", flush=True)
                continue
            r["arranque"] = s
            fits.append(r)
        if not fits:
            continue
        best = max(fits, key=lambda r: r["ll"])
        same = sum(abs(r["ll"] - best["ll"]) < 1e-6 * abs(best["ll"]) for r in fits)
        for r in fits:
            for lab, tv, e, se in zip(labels(spec), truth, r["theta"], r["se"]):
                par.append(dict(rep=rep, estimador=info, arranque=r["arranque"], mejor=r is best,
                                parametro=lab, verdad=tv, estimado=e, se=se, ll=r["ll"],
                                convergio=r["converged"]))
        th_hat = unpack(jnp.asarray(best["x"]), spec, th0, cfg)
        st = market_stats(equilibrium_objects(best["z"], th_hat, cfg), th_hat, cfg)
        row = dict(rep=rep, estimador=info, ll=best["ll"], n_obs=int(cells["cnt"].sum()),
                   n_celdas=len(cells), convergio=best["converged"], n_eval=best["n_eval"],
                   n_fail=best["n_fail"], segundos=best["seconds"], arranques=len(fits),
                   mismo_optimo=same, cond_B=best["cond_B"],
                   P_rmse=precio_rmse(best["z"], z_true, objs_true, cfg))
        row.update({f"{k}_est": v for k, v in st.items()})
        res.append(row)
        z_hat[info] = best["z"]
        print(f"rep {rep} [{info}]: LL = {best['ll']:.2f}, P_rmse = {row['P_rmse']:.3f}, "
              f"convergió = {best['converged']}, {best['seconds']:.0f} s", flush=True)
    return pd.DataFrame(par), pd.DataFrame(res), z_hat


def tabla_precios(z_true, z_hat, cfg):
    J, A, n_act, n = dims(cfg)
    jj, aa = np.meshgrid(np.arange(J), np.arange(1, A), indexing="ij")
    d = dict(j=jj.ravel(), a=aa.ravel(), P_verdad=np.asarray(split_z(z_true, cfg)[1]).ravel())
    d.update({f"P_{k}": np.asarray(split_z(z, cfg)[1]).ravel() for k, z in z_hat.items()})
    return pd.DataFrame(d)


def summarize(outdir):
    p = pd.concat(map(pd.read_csv, glob.glob(os.path.join(outdir, "parametros_reps*.csv"))))
    p = p[p["mejor"]].assign(err=lambda d: d.estimado - d.verdad,
                             cubre=lambda d: (d.estimado - 1.96 * d.se <= d.verdad)
                             & (d.verdad <= d.estimado + 1.96 * d.se))
    g = p.groupby(["estimador", "parametro"], sort=False)
    out = pd.DataFrame(dict(verdad=g.verdad.first(), media=g.estimado.mean(), sesgo=g.err.mean(),
                            sd=g.estimado.std(), se_mediano=g.se.median(),
                            rmse=g.err.apply(lambda e: np.sqrt(np.mean(e ** 2))),
                            cobertura=g.cubre.mean(), reps=g.rep.nunique()))
    out.to_csv(os.path.join(outdir, "resumen_mc.csv"))
    print(out.round(4).to_string())


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0:1", help="réplicas a:b (semillas)")
    ap.add_argument("--N", type=int, default=20_000, help="hogares")
    ap.add_argument("--K", type=int, default=10, help="años por hogar")
    ap.add_argument("--brands", default=",".join(Config.brands))
    ap.add_argument("--types", default=",".join(Config.types))
    ap.add_argument("--a_max", type=int, default=Config.a_max)
    ap.add_argument("--nocar", default=Config.nocar, choices=["reemplaza", "suma"])
    ap.add_argument("--infos", default="parcial,completa")
    ap.add_argument("--fix", default="", help="parámetros fijos en la verdad, p. ej. tc_buy")
    ap.add_argument("--n_starts", type=int, default=2, help="arranques perturbados además de la verdad")
    ap.add_argument("--perturb", type=float, default=0.1)
    ap.add_argument("--lbfgs_iter", type=int, default=500)
    ap.add_argument("--bhhh_iter", type=int, default=50)
    ap.add_argument("--print_every", type=int, default=5, help="línea de avance cada tantas réplicas")
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse()
    if args.smoke:
        args.a_max, args.N, args.K, args.reps = 10, 3_000, 4, "0:1"
        args.n_starts, args.lbfgs_iter, args.bhhh_iter, args.tag = 1, 30, 5, "smoke"
        args.fix, args.verbose = "u1,tc_buy_nocar,tc_sell_inspect,acc_age", True
    args.infos = tuple(args.infos.split(","))
    assert all(i in INFOS for i in args.infos)
    cfg = Config(brands=tuple(args.brands.split(",")), types=tuple(args.types.split(",")),
                 f=tuple([1 / len(args.types.split(","))] * len(args.types.split(","))),
                 a_max=args.a_max, nocar=args.nocar)
    th0 = params.theta(cfg)
    fix = tuple(filter(None, args.fix.split(",")))
    spec = free_spec(th0, fix)
    tag = args.tag or f"A{cfg.a_max}_N{args.N}_K{args.K}_T{cfg.T}_p{len(labels(spec))}"
    outdir = os.path.join(args.outdir, tag)
    if args.summarize:
        return summarize(outdir)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(args), config=dataclasses.asdict(cfg),
                       theta={k: np.asarray(v).tolist() for k, v in th0.items()},
                       libres=labels(spec)), fh, indent=1)

    t0 = time.time()
    z_true, ok, _ = solve(th0, cfg)
    assert ok, "el equilibrio verdadero no converge"
    objs_true = equilibrium_objects(z_true, th0, cfg)
    st = market_stats(objs_true, th0, cfg)
    with open(os.path.join(outdir, "verdad.json"), "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1)
    print(f"equilibrio verdadero ({time.time() - t0:.1f} s): {st}", flush=True)

    a, b = map(int, args.reps.split(":"))
    t_mc = time.time()
    for i, rep in enumerate(range(a, b), start=1):
        par, res, z_hat = una_replica(rep, cfg, th0, spec, z_true, objs_true, args)
        for name, d in (("parametros", par), ("resumen", res)):
            path = os.path.join(outdir, f"{name}_reps{a}-{b - 1}.csv")
            d.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        if rep == a:
            tabla_precios(z_true, z_hat, cfg).to_csv(os.path.join(outdir, "precios.csv"), index=False)
        if i % args.print_every == 0 or rep == b - 1:
            el = time.time() - t_mc
            print(f"[avance] {i}/{b - a} réplicas, {el / 3600:.2f} h transcurridas, "
                  f"~{el / i * (b - a - i) / 3600:.2f} h restantes ({time.strftime('%Y-%m-%d %H:%M')})",
                  flush=True)
    print(f"listo: {outdir}")


if __name__ == "__main__":
    main()
