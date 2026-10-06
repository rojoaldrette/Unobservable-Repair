# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/gen_dataset.py
# Goal:           Regímenes de R, simulación del panel y vistas por escenario de datos
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
Diseño (docs/datasets_montecarlo.md):

- Régimen t = "año" con R_t = R_base * exp(zeta_t).  Cada régimen tiene su propio
  equilibrio estacionario (los agentes creen que R_t es permanente).  Los
  equilibrios son deterministas: se resuelven una vez (`solve_regimes`) y se guardan.
- Panel corto por régimen: N hogares sacados de q_t, K periodos con las matrices del
  régimen t.  Los datos son el equilibrio estacionario de cada régimen, sin
  transiciones entre regímenes.
- Una sola tabla `verdad`; los escenarios (oráculo, ideal, celdas) son vistas.
  El escenario "km" necesita manejo en el modelo (docs/idea_km.md): pendiente.

La simulación es numpy (no jax): es muestreo, no hay nada que diferenciar.
'''

import os
from dataclasses import replace

import numpy as np
import pandas as pd

from utils import dims, decode_states, make_s_grid, split_states, s_new_index
from probabilities import ccps
from primitives import s_transition
from transitions import stationary_q
from ED import solve_equilibrium


# Regímenes ______________________________________________________________

def regime_params(g, zeta):
    R = np.asarray(g.repair_price) * np.exp(zeta)
    return replace(g, repair_price=tuple(tuple(float(x) for x in row) for row in R))


def solve_regimes(g, zetas, path=None, verbose=False):
    # Resuelve el equilibrio de cada régimen (con arranque en caliente) y guarda todo
    # lo que necesita la simulación en un .npz.  Si `path` existe, lo carga.
    if path is not None and os.path.exists(path):
        return load_regimes(path)

    out = dict(zetas=np.asarray(zetas, float))
    keys = ("R", "P", "EV", "q", "keep", "purge", "trade", "buy", "repair", "scrap", "empty", "converged")
    for k in keys:
        out[k] = []
    P = EV = None
    for t, z in enumerate(zetas):
        g_t = regime_params(g, z)
        if verbose:
            print(f"régimen {t}: zeta = {z:+.3f}")
        eq = solve_equilibrium(g_t, P_init=P, EV_init=EV, verbose=verbose)
        c = ccps(eq.EV, eq.P, g_t)
        q = stationary_q(c, g_t)
        vals = dict(R=np.asarray(g_t.repair_price), P=eq.P, EV=eq.EV, q=q, keep=c.keep,
                    purge=c.purge, trade=c.trade, buy=c.buy, repair=c.repair, scrap=c.scrap,
                    empty=eq.empty, converged=eq.converged)
        for k in keys:
            out[k].append(np.asarray(vals[k]))
        P, EV = eq.P, eq.EV
    out = {k: np.stack(v) if isinstance(v, list) else v for k, v in out.items()}
    out["F"] = np.asarray(s_transition(g))            # F[r, j, d, s, s']; R no entra: una sola vez
    if path is not None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, **out)
    return out


def load_regimes(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# Simulación ______________________________________________________________

def _next_state(h, r, cumF, g, rng):
    # Transición física sin la matriz Q (ver transitions.py): la tenencia h pasa a
    # term con prob s (o 1 en la última edad); si sobrevive, s' ~ F_r(. | j, d, s) y edad d+1.
    J, A, S, n_act, n = dims(g)
    grid = np.asarray(make_s_grid(g))
    kh, jh, dh, sh = decode_states(h, g)
    x_next = np.full(len(h), n - 1, dtype=np.int64)            # none -> none
    car = kh < 2
    s_idx = np.where(kh == 0, sh, s_new_index(g)[np.maximum(jh, 0)])   # nuevo: s_new de su marca
    p_term = np.where((kh == 0) & (dh == A - 1), 1.0, grid[s_idx])
    to_term = car & (rng.random(len(h)) < p_term)
    x_next[to_term] = n_act + jh[to_term]
    alive = car & ~to_term
    rows = cumF[r[alive], jh[alive], dh[alive], s_idx[alive]]  # (m, S); nuevo: d = 0, r = 0
    sn = np.minimum((rows < rng.random(alive.sum())[:, None]).sum(axis=1), S - 1)
    x_next[alive] = (jh[alive] * (A - 1) + dh[alive]) * S + sn  # act(j, d+1, s')
    return x_next


def _sample_cat(p, size, rng):
    p = np.clip(p, 0.0, None)
    return rng.choice(len(p), size=size, p=p / p.sum())


DECISION = np.array(["keep", "purge", "trade"])


def simulate_panel(regs, g, N, K, seed):
    # Devuelve la tabla `verdad` (una fila por hogar x régimen x periodo).
    J, A, S, n_act, n = dims(g)
    rng = np.random.default_rng(seed)
    grid = np.asarray(make_s_grid(g))
    cumF = np.cumsum(regs["F"], axis=-1)
    T = len(regs["zetas"])

    frames = []
    car_counter = 0
    for t in range(T):
        keep, purge, trade = regs["keep"][t], regs["purge"][t], regs["trade"][t]
        buy, repair = regs["buy"][t], regs["repair"][t]
        scrap = regs["scrap"][t] if "scrap" in regs else np.zeros_like(keep)
        P = regs["P"][t].ravel()

        x = _sample_cat(regs["q"][t], N, rng)
        car_id = car_counter + np.arange(N)
        car_counter += N
        for k in range(K):
            # Etapa 1
            u = rng.random(N)
            dec = np.where(u < keep[x], 0, np.where(u < keep[x] + purge[x], 1, 2))
            # Chatarreo endógeno: quien se deshace de un coche activo lo chatarrea con prob scrap[x]
            chat = (dec > 0) & (x < n_act) & (rng.random(N) < scrap[x])
            h = np.where(dec == 0, x, n - 1)
            is_trade = dec == 2
            h[is_trade] = _sample_cat(buy, is_trade.sum(), rng)
            car_id = np.where(is_trade, car_counter + np.arange(N), car_id)
            car_counter += N
            # Etapa 2
            r = (rng.random(N) < repair[h]).astype(np.int8)
            # Transición física
            x_next = _next_state(h, r, cumF, g, rng)

            kx, jx, ax, sx = decode_states(x, g)
            kh, jh, dh, sh = decode_states(h, g)
            kn, jn, an, sn = decode_states(x_next, g)
            has_car = kh < 2
            to_term = has_car & (kn == 1)
            end_life = to_term & (kh == 0) & (dh == A - 1)
            # s de la tenencia: usado -> su s; nuevo -> s_new en el grid
            sh_idx = np.where(kh == 0, sh, np.where(kh == 1, s_new_index(g)[np.maximum(jh, 0)], -1))

            frames.append(pd.DataFrame({
                "id_hogar": np.arange(N), "t": t, "k": k, "id_coche": np.where(has_car, car_id, -1),
                "x": x, "h": h, "x_next": x_next,                # índices (layouts X, H, X)
                "chatarreo": chat.astype(np.int8),               # se deshizo del coche activo chatarreándolo
                "estado": np.array(["activo", "terminal", "sin_coche"])[kx],
                "j": jx, "a": ax, "s_idx": sx,
                "decision": np.where((kx == 2) & (dec == 1), "stay_none", DECISION[dec]),
                "tipo_h": np.array(["usado", "nuevo", "sin_coche"])[kh],
                "j_h": jh, "d_h": dh, "s_h_idx": sh_idx,
                "r": r,
                "p_rep": repair[h], "p_keep": keep[x], "p_trade": trade[x],
                "accidente": (to_term & ~end_life).astype(np.int8),
                "fin_vida": end_life.astype(np.int8),
                "s_next_idx": np.where(kn == 0, sn, -1),
                "P_h": np.where(kh == 0, P[np.minimum(h, n_act - 1)], np.nan),
            }))
            car_id = np.where(has_car & (kn == 0), car_id, -1)
            x = x_next

    df = pd.concat(frames, ignore_index=True)
    df["s_h"] = np.where(df["s_h_idx"] >= 0, grid[df["s_h_idx"].clip(0)], np.nan)
    df["s_next"] = np.where(df["s_next_idx"] >= 0, grid[df["s_next_idx"].clip(0)], np.nan)
    return df


def regs_from_objects(objs, F):
    # Para simulate_panel: objs = lista por régimen de dicts de equilibrium.equilibrium_objects
    # de UN tipo (P, q, keep, purge, trade, buy, repair).  F = s_transition(g).
    regs = {k: np.stack([o[k] for o in objs]) for k in ("P", "q", "keep", "purge", "trade", "buy", "repair", "scrap")}
    regs["zetas"] = np.zeros(len(objs))
    regs["F"] = np.asarray(F)
    return regs


def simulate_economy(objs, F, g, f, N, K, seed):
    # Varios tipos de hogar.  objs[t][tau] = objetos del régimen t y el tipo tau.
    # N hogares por régimen en total, repartidos según f.  Agrega la columna `tipo` e
    # ids de hogar únicos por (régimen, hogar).
    frames, start = [], 0
    for tau, ft in enumerate(f):
        Nt = int(round(N * ft))
        regs = regs_from_objects([objs_t[tau] for objs_t in objs], F)
        d = simulate_panel(regs, g, Nt, K, seed=1_000 * seed + tau)
        d.insert(1, "tipo", tau)
        d["id_hogar"] = d["id_hogar"] + start
        start += Nt
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["id_hogar"] = df["t"] * start + df["id_hogar"]          # hogares distintos en cada régimen
    return df


# Vistas por escenario ______________________________________________________________

HIDDEN = ["r", "p_rep", "p_keep", "p_trade", "P_h"]


def view(df, escenario):
    if escenario == "oraculo":
        return df
    if escenario == "ideal":
        return df.drop(columns=HIDDEN)
    if escenario == "km":
        raise NotImplementedError("el modelo no tiene manejo todavía (docs/idea_km.md)")
    raise ValueError(escenario)


def to_cells(df, g):
    # Como los datos daneses: conteos por celda, sin s ni r.
    # decisiones: por (t, j, a) del coche al inicio; salidas y compras: por (t, j_h, d_h).
    act = df[df["estado"] == "activo"]
    dec = (act.groupby(["t", "j", "a"])["decision"].value_counts().unstack(fill_value=0)
              .reindex(columns=["keep", "purge", "trade"], fill_value=0)
              .add_prefix("n_").reset_index())
    dec["n"] = dec[["n_keep", "n_purge", "n_trade"]].sum(axis=1)

    hold = df[df["tipo_h"] != "sin_coche"]
    scrap = (hold.groupby(["t", "j_h", "d_h"])
                 .agg(n_hold=("accidente", "size"), n_scrap=("accidente", "sum"),
                      n_fin_vida=("fin_vida", "sum"))
                 .reset_index())
    bought = df[df["decision"] == "trade"]
    buys = bought.groupby(["t", "j_h", "d_h"]).size().rename("n_buy").reset_index()
    return dict(decisiones=dec, salidas=scrap, compras=buys)


def gen_dataset(regs, g, seed, N=20_000, K=2, escenario="ideal"):
    # Al estilo de scripts/main/gen_dataset.py: devuelve el DataFrame del escenario
    # (o un dict de DataFrames si escenario == "celdas").
    df = simulate_panel(regs, g, N, K, seed)
    if escenario == "celdas":
        return to_cells(df, g)
    return view(df, escenario)
