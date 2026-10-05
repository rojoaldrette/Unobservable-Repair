# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/montecarlo.py
# Goal:           Monte Carlo del estimador DNFXP de Gillingham
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           04/10/2026
#
# _____________________________________________________________________________
'''
El paper no reporta un Monte Carlo (estima con datos daneses).  Este replica su
estimador sobre datos simulados del propio modelo:

1. `prepare`: resuelve el equilibrio verdadero (theta_0 = GParams) y lo guarda.
2. `one_mc(rep)`: simula un panel (N hogares, K años) con semilla rep, lo agrega a
   celdas y estima por máxima verosimilitud con:
     - "parcial":  como el paper (sec. 5.1, apéndice D): sin precios ni accidentes;
                   un coche que sale del parque puede ser accidente o chatarreo endógeno.
     - "completa": oráculo, ve el estado verdadero (accidentes incluidos).
   Varios arranques (la verdad + n_starts perturbados); se queda el de mayor LL.
3. Además de theta, compara objetos de equilibrio en theta_hat contra los verdaderos:
   precios P (que el econometrista nunca ve), tasa de chatarreo endógeno, tasa de
   accidentes y la fracción de salidas que es voluntaria.

Tipos de hogar (GTypes): mu, u0, u1, tc_buy y tc_buy_nocar por tipo; Ts, sigma_sell y
accidentes comunes (como el paper).  Un tipo = el diseño original.
'''

import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import jax.numpy as jnp

from utils import dims, split_states
from bellman import accident_prob
from params import GTypes
from theta import FIELDS, economy, theta_types, free_spec, pack, unpack, natural, labels, Economy
from equilibrium import solve, split_z
from simulate import equilibrium_objects, simulate_panel, to_cells
from loglikelihood import treat_data, LLEval, estimate as estim_ml

ABBR = {"low_couple_poor": "lcp", "low_couple_rich": "lcr",
        "low_single_poor": "lsp", "low_single_rich": "lsr"}


@dataclass(frozen=True)
class MCDesign:
    N: int = 20_000            # hogares (en total, repartidos según f)
    K: int = 10                # años por hogar (el paper: 1997-2008)
    infos: tuple = ("parcial", "completa")
    free: tuple = FIELDS       # parámetros que se estiman (el resto se fija en la verdad)
    n_starts: int = 2          # arranques perturbados además de la verdad
    perturb: float = 0.1       # x_start = x_0 + perturb |x_0| N(0, 1)
    types: GTypes = GTypes()

    def tag(self, g):
        th = theta_types(g, self.types)
        nfree = sum(int(np.prod(s)) for _, s in free_spec(th, self.free))
        tag = f"N{self.N}_K{self.K}_A{g.a_max}_p{nfree}"
        if len(self.types.names) > 1:
            tag += "_T" + "-".join(ABBR.get(k, k) for k in self.types.names)
        return tag


# Objetos de equilibrio ______________________________________________________________

def market_stats(z, eco):
    # Tasas agregadas sobre tipos (ponderadas por f), por coche en circulación
    g = eco.g
    J, A, n_act, n = dims(g)
    eqos = equilibrium_objects(z, eco)
    al = accident_prob(eco.type_model(0))           # accidentes comunes a los tipos
    endo = acc = fin = cars = none = 0.0
    qa_all = 0.0
    for e, ft in zip(eqos, eco.f):
        q = e["q"]
        qa, _, qn = split_states(jnp.asarray(q), g)
        keep_a, _, _ = split_states(jnp.asarray(e["keep"]), g)
        scrap_a, _, _ = split_states(jnp.asarray(e["scrap_x"]), g)
        endo += ft * float(jnp.sum(qa * (1 - keep_a) * scrap_a))
        # Tenencias: keep se queda en x, trade compra según buy, purge va a "sin coche"
        qH = q * e["keep"] + float(q @ e["trade"]) * e["buy"]
        hu, hn, _ = split_states(jnp.asarray(qH), g)
        acc += ft * float(jnp.sum(hu[:, :A - 2] * al[:, 1:A - 1]) + jnp.sum(hn * al[:, 0]))
        fin += ft * float(jnp.sum(hu[:, A - 2]))
        cars += ft * float(jnp.sum(hu) + jnp.sum(hn))
        none += ft * float(qn)
        qa_all = qa_all + ft * np.asarray(qa)
    stats = dict(sin_coche=none, endo_rate=endo / cars, acc_rate=acc / cars,
                 share_endo=endo / (endo + acc + fin))
    return stats, dict(P=np.asarray(split_z(z, eco)[1]), qa=qa_all, eqos=eqos)


def prepare(g, design, outdir, verbose=False):
    eco = economy(g, design.types)
    z, ok, it = solve(eco, verbose=verbose)
    if not ok:
        raise RuntimeError("el equilibrio verdadero no converge")
    stats, e = market_stats(z, eco)
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, f"truth_{design.tag(g)}.npz"), z=np.asarray(z),
                        P=e["P"], qa=e["qa"])
    if verbose:
        print("equilibrio verdadero:", {k: round(v, 4) for k, v in stats.items()})
    return z, e, stats


# Una réplica ______________________________________________________________

def estimate(g, design, cells, info, z_true, rng, verbose=False):
    # Arranques: la verdad (con el equilibrio verdadero como arranque en caliente) y
    # n_starts perturbados.  Se queda el de mayor LL.
    th0 = theta_types(g, design.types)
    spec = free_spec(th0, design.free)
    x0 = np.asarray(pack(th0, spec))
    data = treat_data(cells, info)
    starts = [x0] + [x0 + design.perturb * np.abs(x0) * rng.standard_normal(len(x0))
                     for _ in range(design.n_starts)]
    fits = []
    for s, xs in enumerate(starts):
        ev = LLEval(g, design.types.f, spec, th0, data, info, z0=z_true if s == 0 else None)
        try:
            fits.append(estim_ml(ev, xs, verbose=verbose))
        except (RuntimeError, np.linalg.LinAlgError) as err:
            if verbose:
                print(f"  arranque {s} falló: {err}")
    best = max(fits, key=lambda r: r["ll"])
    # ¿Los arranques perturbados llegan al mismo óptimo? (multiplicidad de máximos locales)
    best["n_fits"] = len(fits)
    best["n_same"] = sum(abs(r["ll"] - best["ll"]) < 1e-4 for r in fits)
    best["eco"] = Economy(g, unpack(jnp.asarray(best["x"]), spec, th0), design.types.f)
    return best


def one_mc(rep, z_true, e0, g, design, verbose=False):
    t0 = time.time()
    rng = np.random.default_rng(10_000 + rep)
    df = simulate_panel(e0["eqos"], g, design.types.f, design.N, design.K, seed=rep)
    cells = to_cells(df)

    th0 = theta_types(g, design.types)
    spec = free_spec(th0, design.free)
    truth = np.asarray(natural(jnp.asarray(pack(th0, spec)), spec, th0))
    qa = np.ravel(e0["qa"])
    P0 = np.ravel(e0["P"])

    row = dict(rep=rep, n_obs=int(cells["cnt"].sum()), n_cells=len(cells),
               sim_accidentes=int(df["accidente"].sum()),
               sim_chatarreo_endogeno=int(df["chatarreo_endogeno"].sum()))
    for info in design.infos:
        r = estimate(g, design, cells, info, z_true, rng, verbose=verbose)
        for lab, t, e, s in zip(labels(spec), truth, r["theta"], r["se"]):
            row[f"{info}_{lab}"] = e
            row[f"{info}_se_{lab}"] = s
            row[f"{info}_cover_{lab}"] = float(abs(e - t) <= 1.96 * s) if np.isfinite(s) else np.nan
        row[f"{info}_ll"], row[f"{info}_ok"] = r["ll"], r["converged"]
        row[f"{info}_iters"], row[f"{info}_seconds"] = r["iters"], r["seconds"]
        row[f"{info}_starts_same_opt"] = f"{r['n_same']}/{r['n_fits']}"
        stats, e = market_stats(r["z"], r["eco"])
        for k, v in stats.items():
            row[f"{info}_{k}"] = v
        dP = np.ravel(e["P"]) - P0
        row[f"{info}_P_rmse"] = float(np.sqrt(np.sum(qa * dP ** 2) / np.sum(qa)))
        row[f"{info}_P_maxabs"] = float(np.max(np.abs(dP)))
    row["seconds"] = time.time() - t0
    return row


def montecarlo(g, design, reps, outdir, verbose=False):
    z_true, e0, stats0 = prepare(g, design, outdir, verbose=verbose)
    th0 = theta_types(g, design.types)
    spec = free_spec(th0, design.free)
    truth = np.asarray(natural(jnp.asarray(pack(th0, spec)), spec, th0))
    true_row = {f"true_{l}": v for l, v in zip(labels(spec), truth)}
    true_row.update({f"true_{k}": v for k, v in stats0.items()})
    out = os.path.join(outdir, f"mc_{design.tag(g)}_reps{reps[0]}-{reps[-1]}.csv")
    rows = []
    for rep in reps:
        row = one_mc(rep, z_true, e0, g, design, verbose=verbose)
        d = asdict(design)
        d["infos"], d["free"] = ",".join(d["infos"]), ",".join(d["free"])
        d["types"] = ",".join(design.types.names)
        d["f"] = ",".join(str(v) for v in design.types.f)
        rows.append({**row, **true_row, **d})
        pd.DataFrame(rows).to_csv(out, index=False)
        if verbose:
            print(f"rep {rep}: {row['seconds']:.0f} s, " + ", ".join(
                f"{i}: LL {row[f'{i}_ll']:.1f} P_rmse {row[f'{i}_P_rmse']:.2f}"
                for i in design.infos if f"{i}_ll" in row))
    return pd.DataFrame(rows)


def summarize(df, g, design):
    # Tabla de sesgo, RMSE, se medio y cobertura por parámetro y estimador
    th0 = theta_types(g, design.types)
    spec = free_spec(th0, design.free)
    labs = labels(spec)
    rows = []
    for info in design.infos:
        for lab in labs + ["sin_coche", "endo_rate", "acc_rate", "share_endo"]:
            t = df[f"true_{lab}"].iloc[0]
            e = df[f"{info}_{lab}"]
            r = dict(estimador=info, parametro=lab, verdad=t, media=e.mean(),
                     sesgo=e.mean() - t, rmse=np.sqrt(((e - t) ** 2).mean()))
            if f"{info}_se_{lab}" in df:
                r["se_medio"] = df[f"{info}_se_{lab}"].mean()
                r["sd_mc"] = e.std()
                r["cobertura_95"] = df[f"{info}_cover_{lab}"].mean()
            rows.append(r)
        rows.append(dict(estimador=info, parametro="P_rmse (miles DKK)", media=df[f"{info}_P_rmse"].mean()))
        # Dirección casi no identificada: corr(tc_buy, tc_sell) entre réplicas
        for lab in labs:
            if lab.startswith("tc_buy_t") and "tc_sell" in labs:
                rows.append(dict(estimador=info, parametro=f"corr({lab}, tc_sell)",
                                 media=df[f"{info}_{lab}"].corr(df[f"{info}_tc_sell"])))
    return pd.DataFrame(rows)
