# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/modelo_tesis/montecarlo.py
# Goal:           Monte Carlo: 3 diseños de modelo_tesis y Gillingham sobre el mismo panel
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
Uso (desde claude/niu/modelo_tesis/):

    python -u montecarlo.py --smoke                     # juguete (CPU, minutos)
    python -u montecarlo.py --reps 0:1 -v               # una réplica completa (GPU), con log de L-BFGS
    CUDA_VISIBLE_DEVICES=0 python -u montecarlo.py --reps 0:25  > mc_0-24.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python -u montecarlo.py --reps 25:50 > mc_25-49.log 2>&1 &
    python -u montecarlo.py --summarize                 # tabla de sesgo, RMSE y cobertura

Por réplica (semilla = rep), modelo_fin.md secs. 6-7:
1. Simula el panel: N hogares por régimen, K años, 3 regímenes de R.
2. Lanza en paralelo, en CPU, la estimación de Gillingham sobre el mismo panel sin w ni
   reparación (niu/gillingham/estimar_panel.py; verosimilitudes "parcial" y "completa").
3. En la GPU estima modelo_tesis con los diseños 1 (ve r y el motivo de salida), 2 (no ve r)
   y 3 (no ve r ni el motivo).  Arranque: la verdad; en la réplica --rep_perturbados, además
   --n_perturbados arranques perturbados (x0 + 0.1 |x0| N(0, 1)).
4. Escribe los CSV de la réplica e imprime una línea; cada --print_every réplicas, [avance].

Salidas en claude/niu/output/modelo_tesis/montecarlo/<tag>/:
    config.json
    parametros_reps<a>-<b>.csv       rep, estimador (diseno_1/2/3), arranque, mejor, parametro, verdad,
                                     estimado, se, ll, convergio
    resumen_reps<a>-<b>.csv          rep, estimador, ll, n_obs, n_celdas, convergio, n_eval, n_fail, cuerda,
                                     refact, newton, segundos, cond_B, P_rmse, reparacion_verdad, reparacion_est
    gill_parametros_reps<a>-<b>.csv  lo mismo para Gillingham (estimador gill_parcial / gill_completa)
    gill_resumen_reps<a>-<b>.csv
    resumen_mc.csv                   (--summarize) por estimador y parámetro: sesgo, sd, se mediano,
                                     RMSE, cobertura al 95%
'''

import argparse
import dataclasses
import glob
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from params import Config, theta, regime_theta
from utils import dims, split_states
from equilibrio import solve_regimes, factor, equilibrium_objects, market_stats, split_z
from gen_dataset import simulate_panel, decode
from ll_estim import (free_spec, pack, unpack, natural, labels, to_cells, cell_data, LLEval,
                      estimate, DESIGNS)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "output", "modelo_tesis"))
GILL = os.path.normpath(os.path.join(HERE, "..", "gillingham"))


# Verdad ______________________________________________________________

def true_equilibria(th, cfg, desde, verbose):
    # Equilibrios de teoria/<desde>/ si son del mismo θ y tamaño; si no, se resuelven
    J, A, W, n_act, n = dims(cfg)
    d = os.path.join(OUT, "teoria", desde)
    try:
        zs = np.load(os.path.join(d, "equilibrios.npz"))["zs"]
        with open(os.path.join(d, "config.json"), encoding="utf-8") as fh:
            th_saved = json.load(fh)["theta"]
        same = (zs.shape == (len(cfg.zetas), cfg.T * n + n_act)
                and all(np.allclose(np.asarray(th_saved[k]), np.asarray(v)) for k, v in th.items()))
    except (OSError, KeyError):
        same = False
    if same:
        print(f"equilibrios verdaderos de {d}", flush=True)
        return [jnp.asarray(z) for z in zs]
    print("resolviendo los equilibrios verdaderos", flush=True)
    return solve_regimes(th, cfg, verbose=verbose)[0]


def to_gillingham(df, cfg):
    # Panel en el espacio de estados de Gillingham: (j, a) sin w; sin r
    J, A, W, n_act, n = dims(cfg)
    nA, ng = J * (A - 1), J * (A - 1) + J + 1
    jx, ax, _ = decode(df["x"].to_numpy(), cfg, "X")
    jh, dh, _ = decode(df["h"].to_numpy(), cfg, "H")
    x = np.where(ax >= 1, jx * (A - 1) + ax - 1, np.where(ax == 0, nA + jx, ng - 1))
    h = np.where(dh >= 1, jh * (A - 1) + dh - 1, np.where(dh == 0, nA + jh, ng - 1))
    return pd.DataFrame(dict(id=df["id"], tipo=df["tipo"], k=df["k"], x=x, o=df["o"], h=h))


def launch_gillingham(df, cfg, args, rep, outdir, bloque):
    path = os.path.join(outdir, f"tmp_panel_gill_rep{rep}.csv.gz")
    to_gillingham(df, cfg).to_csv(path, index=False)
    cmd = [sys.executable, "-u", "estimar_panel.py", "--panel", path, "--outdir", outdir,
           "--rep", str(rep), "--bloque", bloque, "--a_max", str(cfg.a_max),
           "--brands", ",".join(cfg.brands), "--types", ",".join(cfg.types), "--nocar", cfg.nocar]
    if args.smoke:
        cmd += ["--lbfgs_iter", "20", "--bhhh_iter", "3"]
    return subprocess.Popen(cmd, cwd=GILL, env=dict(os.environ, JAX_PLATFORMS="cpu")), path


# Una réplica ______________________________________________________________

def una_replica(rep, cfg, th0, spec, zs_true, Jinvs_true, objs_true, args, outdir, bloque):
    t_rep = time.time()
    df = simulate_panel(objs_true, cfg, args.N, args.K, seed=rep)
    gill = launch_gillingham(df, cfg, args, rep, outdir, bloque) if args.gillingham else None

    x0 = np.asarray(pack(th0, spec, cfg))
    truth = np.asarray(natural(jnp.asarray(x0), spec, th0, cfg))
    rng = np.random.default_rng(10_000 + rep)
    starts = [x0]
    if rep == args.rep_perturbados:
        starts += [x0 + 0.1 * np.abs(x0) * rng.standard_normal(len(x0)) for _ in range(args.n_perturbados)]
    w_true = [sum(f * split_states(o["q"], cfg)[0] for o, f in zip(objs, cfg.f)) for objs in objs_true]
    rep_true = [market_stats(objs, cfg)["reparacion"] for objs in objs_true]

    par, res = [], []
    for design in args.designs:
        cells = to_cells(df, cfg, design)
        data = cell_data(cells, cfg)
        fits = []
        for s, xs in enumerate(starts):
            if args.verbose:
                print(f"  rep {rep} [diseño {design}] arranque {s}", flush=True)
            ev = LLEval(cfg, spec, th0, data, design, zs_true, Jinvs_true, x0)
            try:
                r = estimate(ev, xs, args.lbfgs_iter, args.bhhh_iter, verbose=args.verbose)
            except RuntimeError as err:
                print(f"  rep {rep} [diseño {design}] arranque {s} falló: {err}", flush=True)
                continue
            r["arranque"] = s
            fits.append(r)
        if not fits:
            continue
        best = max(fits, key=lambda r: r["ll"])
        for r in fits:
            for lab, tv, e, se in zip(labels(spec), truth, r["theta"], r["se"]):
                par.append(dict(rep=rep, estimador=f"diseno_{design}", arranque=r["arranque"],
                                mejor=r is best, parametro=lab, verdad=tv, estimado=e, se=se,
                                ll=r["ll"], convergio=r["converged"]))
        th_hat = unpack(jnp.asarray(best["x"]), spec, th0, cfg)
        num = den = 0.0
        rep_est = []
        for t, z in enumerate(best["zs"]):
            dP = np.asarray(split_z(z, cfg)[1] - split_z(zs_true[t], cfg)[1])
            num += float((w_true[t] * dP ** 2).sum())
            den += float(w_true[t].sum())
            rep_est.append(market_stats(equilibrium_objects(z, regime_theta(th_hat, cfg, t), cfg), cfg)["reparacion"])
        res.append(dict(rep=rep, estimador=f"diseno_{design}", ll=best["ll"],
                        n_obs=int(cells["cnt"].sum()), n_celdas=len(cells), convergio=best["converged"],
                        **{k: best[k] for k in ("n_eval", "n_fail", "cuerda", "refact", "newton", "cond_B")},
                        segundos=best["seconds"], arranques=len(fits),
                        P_rmse=float(np.sqrt(num / den)),
                        reparacion_verdad=float(np.mean(rep_true)), reparacion_est=float(np.mean(rep_est))))
        print(f"rep {rep} [diseño {design}]: LL = {best['ll']:.2f}, convergió = {best['converged']}, "
              f"{best['n_eval']} evals, {best['seconds']:.0f} s", flush=True)

    for name, d in (("parametros", par), ("resumen", res)):
        path = os.path.join(outdir, f"{name}{bloque}.csv")
        pd.DataFrame(d).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    if gill is not None:
        proc, path = gill
        if proc.wait() != 0:
            print(f"  OJO: falló Gillingham en la réplica {rep}", flush=True)
        os.remove(path)
    print(f"rep {rep} lista ({time.time() - t_rep:.0f} s)", flush=True)


# Resumen ______________________________________________________________

def summarize(outdir):
    files = glob.glob(os.path.join(outdir, "parametros_reps*.csv")) + glob.glob(os.path.join(outdir, "gill_parametros_reps*.csv"))
    p = pd.concat(map(pd.read_csv, files))
    p = p[p["mejor"]].assign(err=lambda d: d.estimado - d.verdad,
                             cubre=lambda d: (d.estimado - 1.96 * d.se <= d.verdad) & (d.verdad <= d.estimado + 1.96 * d.se))
    g = p.groupby(["estimador", "parametro"], sort=False)
    out = pd.DataFrame(dict(verdad=g.verdad.first(), media=g.estimado.mean(), sesgo=g.err.mean(),
                            sesgo_rel=g.err.mean() / g.verdad.first().abs(), sd=g.estimado.std(),
                            se_mediano=g.se.median(), rmse=g.err.apply(lambda e: np.sqrt(np.mean(e ** 2))),
                            cobertura=g.cubre.mean(), reps=g.rep.nunique()))
    out.to_csv(os.path.join(outdir, "resumen_mc.csv"))
    print(out.round(4).to_string())


# Main ______________________________________________________________

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0:1")
    ap.add_argument("--N", type=int, default=10_000, help="hogares por régimen")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--designs", default="1,2,3")
    ap.add_argument("--sin_gillingham", dest="gillingham", action="store_false")
    ap.add_argument("--rep_perturbados", type=int, default=0)
    ap.add_argument("--n_perturbados", type=int, default=2)
    ap.add_argument("--lbfgs_iter", type=int, default=300)
    ap.add_argument("--bhhh_iter", type=int, default=30)
    ap.add_argument("--print_every", type=int, default=5)
    ap.add_argument("--desde", default="R_LB", help="tag de teoria/ con los equilibrios verdaderos")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse()
    if args.smoke:
        cfg = Config(a_max=8, w_step=0.15, w_min=-1.5, w_max=1.5, jac_chunk=128)
        args.N, args.K, args.reps, args.tag = 2_000, 3, "0:1", "smoke"
        args.lbfgs_iter, args.bhhh_iter, args.n_perturbados, args.verbose = 20, 3, 1, True
    else:
        cfg = Config()
    args.designs = tuple(int(d) for d in args.designs.split(","))
    assert all(d in DESIGNS for d in args.designs)
    outdir = os.path.join(OUT, "montecarlo", args.tag)
    if args.summarize:
        return summarize(outdir)
    os.makedirs(outdir, exist_ok=True)
    th0 = theta(cfg)
    spec = free_spec(th0)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(args), config=dataclasses.asdict(cfg),
                       theta={k: np.asarray(v).tolist() for k, v in th0.items()}, libres=labels(spec)), fh, indent=1)
    print(f"dispositivo: {jax.devices()[0]} | {len(labels(spec))} parámetros libres | diseños {args.designs}"
          f" | Gillingham: {args.gillingham}", flush=True)

    t0 = time.time()
    zs_true = true_equilibria(th0, cfg, args.desde if not args.smoke else "-", args.verbose)
    ths = [regime_theta(th0, cfg, t) for t in range(len(cfg.zetas))]
    Jinvs_true = [factor(z, th_, cfg)[1] for z, th_ in zip(zs_true, ths)]
    objs_true = [equilibrium_objects(z, th_, cfg) for z, th_ in zip(zs_true, ths)]
    print(f"verdad lista ({time.time() - t0:.0f} s)", flush=True)

    a, b = map(int, args.reps.split(":"))
    bloque = f"_reps{a}-{b - 1}"
    t_mc = time.time()
    for i, rep in enumerate(range(a, b), start=1):
        una_replica(rep, cfg, th0, spec, zs_true, Jinvs_true, objs_true, args, outdir, bloque)
        if i % args.print_every == 0 or rep == b - 1:
            el = time.time() - t_mc
            print(f"[avance] {i}/{b - a} réplicas, {el / 3600:.2f} h transcurridas, "
                  f"~{el / i * (b - a - i) / 3600:.2f} h restantes ({time.strftime('%Y-%m-%d %H:%M')})", flush=True)
    print(f"listo: {outdir}")


if __name__ == "__main__":
    main()
