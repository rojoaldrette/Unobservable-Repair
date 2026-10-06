# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/montecarlo.py
# Goal:           Monte Carlo: oráculo vs ingenuo vs Hu & Xin
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
Flujo:
1. `prepare`: resuelve los T equilibrios (uno por régimen de R) y los guarda en .npz.
   Es la parte cara y determinista: se hace una vez por diseño.
2. `one_mc(rep)`: simula el panel con semilla rep, estima oráculo / ingenuo / hx y
   devuelve una fila con estimaciones y métricas de recuperación de Pr(repair | h).
3. `montecarlo`: corre un bloque de réplicas y va escribiendo un CSV (si el job se
   cae, lo hecho queda guardado).  Pensado para arrays de SLURM (ver main.py).
'''

import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import jax.numpy as jnp

from gen_dataset import solve_regimes, simulate_panel, view
from loglikelihood import (treat_data, start_values, estim_ll, summarize_theta,
                           true_theta, p_repair)
from utils import make_state_grid


@dataclass(frozen=True)
class MCDesign:
    N: int = 20_000          # hogares por régimen
    K: int = 2               # periodos simulados por hogar dentro de un régimen
    T: int = 13              # regímenes ("años")
    spread: float = 0.3      # zeta_t en [-spread, spread]: R_t = R_base * exp(zeta_t)
    spec: str = "flexible"   # especificación de p en hx: "flexible" | "logit_R"
    se: bool = False         # errores estándar por hessiano (para cobertura)

    def zetas(self):
        return np.linspace(-self.spread, self.spread, self.T)

    def regimes_tag(self, g):
        return f"T{self.T}_sp{self.spread}_ns{g.n_s}_A{g.a_max}"

    def tag(self, g):
        return f"N{self.N}_K{self.K}_{self.regimes_tag(g)}_{self.spec}"


def prepare(g, design, outdir, verbose=False):
    path = os.path.join(outdir, f"regimes_{design.regimes_tag(g)}.npz")
    t0 = time.time()
    regs = solve_regimes(g, design.zetas(), path=path, verbose=verbose)
    if verbose:
        print(f"regímenes listos ({time.time() - t0:.0f} s): {path}; "
              f"convergieron {int(np.sum(regs['converged']))}/{design.T}")
    return regs


def _eval_points(df, g, R):
    # Todas las tenencias usadas con información (d <= A-2), agregadas por celda,
    # con la Pr(repair) verdadera: ahí se mide qué tan bien se recupera p_t(h).
    A = g.a_max
    d = df[(df["tipo_h"] == "usado") & (df["d_h"] <= A - 2)]
    cell = (d.groupby(["t", "j_h", "d_h", "s_h_idx"])
             .agg(cnt=("p_rep", "size"), p_true=("p_rep", "mean"), r_bar=("r", "mean"))
             .reset_index())
    grid = np.asarray(make_state_grid(g))
    data = dict(t=jnp.asarray(cell["t"].to_numpy()), j=jnp.asarray(cell["j_h"].to_numpy()),
                d=jnp.asarray(cell["d_h"].to_numpy()), s=jnp.asarray(grid[cell["s_h_idx"].to_numpy()]),
                new=jnp.zeros(len(cell), bool),
                R=jnp.asarray(R[cell["t"].to_numpy(), cell["j_h"].to_numpy(), cell["d_h"].to_numpy() - 1]))
    return cell, data


def one_mc(rep, regs, g, design, verbose=False):
    t0 = time.time()
    df = simulate_panel(regs, g, design.N, design.K, seed=rep)
    data_r = treat_data(df, g, with_r=True)
    data = treat_data(view(df, "ideal"), g, R=regs["R"])

    est_o = estim_ll("oracle", data_r, start_values(g, "oracle"), se=design.se, verbose=verbose)
    est_n = estim_ll("naive", data, start_values(g, "naive"), se=design.se, verbose=verbose)
    th0 = start_values(g, "hx", design.spec, n_regimes=design.T)
    for k in ("c", "s_age", "s_persist", "log_sd"):          # arranca desde el ingenuo
        th0[k] = est_n["theta"][k]
    est_h = estim_ll("hx", data, th0, spec=design.spec, se=design.se, verbose=verbose)

    row = dict(rep=rep, n_obs=int(jnp.sum(data["cnt"])), share_rep=float(df["r"].mean()))
    for name, est in (("oracle", est_o), ("naive", est_n), ("hx", est_h)):
        for k, v in summarize_theta(est["theta"]).items():
            row[f"{name}_{k}"] = v
        row[f"{name}_ll"], row[f"{name}_ok"] = est["ll"], est["success"]
        if design.se and "se" in est:
            se = est["se"]
            row[f"{name}_se_s_age"] = float(se["s_age"])
            if "log_srep" in se:   # delta method: se(s_repair) = s_repair * se(log s_repair)
                row[f"{name}_se_s_repair"] = row[f"{name}_s_repair"] * float(se["log_srep"])

    # Recuperación de Pr(repair | h, t)
    cell, ev = _eval_points(df, g, regs["R"])
    p_hat = np.asarray(p_repair(est_h["theta"], ev, design.spec))
    w = cell["cnt"].to_numpy() / cell["cnt"].sum()
    err = p_hat - cell["p_true"].to_numpy()
    row["hx_p_bias"] = float(np.sum(w * err))
    row["hx_p_rmse"] = float(np.sqrt(np.sum(w * err ** 2)))
    for a in sorted(cell["d_h"].unique()):
        m = (cell["d_h"] == a).to_numpy()
        row[f"hx_p_rmse_d{a}"] = float(np.sqrt(np.sum(w[m] * err[m] ** 2) / np.sum(w[m])))
    row["seconds"] = time.time() - t0
    return row


def montecarlo(g, design, reps, outdir, verbose=False):
    os.makedirs(outdir, exist_ok=True)
    regs = prepare(g, design, outdir, verbose=verbose)
    out = os.path.join(outdir, f"mc_{design.tag(g)}_reps{reps[0]}-{reps[-1]}.csv")
    truth = {f"true_{k}": v for k, v in true_theta(g).items()}
    rows = []
    for rep in reps:
        row = one_mc(rep, regs, g, design, verbose=verbose)
        rows.append({**row, **truth, **asdict(design)})
        pd.DataFrame(rows).to_csv(out, index=False)
        if verbose:
            print(f"rep {rep}: s_repair hx = {row['hx_s_repair']:.4f} (verdad {g.s_repair}), "
                  f"RMSE p = {row['hx_p_rmse']:.4f}, {row['seconds']:.0f} s")
    return pd.DataFrame(rows)
