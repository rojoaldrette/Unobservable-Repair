# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/simulate.py
# Goal:           Panel simulado de hogares desde el equilibrio estacionario
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           04/10/2026
#
# _____________________________________________________________________________
'''
Como el registro danés: N hogares seguidos K periodos, con el estado inicial sacado de
la distribución estacionaria q (sec. 5.1: los datos se generan en el estado
estacionario).  Cada periodo:

    x  (inicio)  ->  decisión o en {keep, purge, trade} y, si se deshace de un coche
                     activo, vender o chatarrear (sigma_sell; en terminal: chatarra)
                 ->  tenencia h  ->  x' ~ Q[h]   (accidente con prob alpha(j, d))

Códigos de resultado `o` (lo que usa la verosimilitud):
    0 keep | 1 purge, coche vendido | 2 purge, coche chatarreado
           | 3 trade, coche vendido | 4 trade, coche chatarreado
"Chatarreado" junta el chatarreo endógeno, el accidente y el fin de vida: en los datos
solo se ve que el coche salió del parque.  Desde "sin coche": purge = seguir sin coche
(o = 1) y trade = comprar (o = 3).

La tabla trae también lo que el econometrista no ve (`accidente`, `chatarreo_endogeno`,
el estado x verdadero), para el estimador de información completa.

Con T tipos de hogar: round(N f_t) hogares de cada tipo, cada uno con las CCPs de su
tipo y los mismos precios.  El tipo se observa (como en el paper).
'''

import numpy as np
import pandas as pd
import jax.numpy as jnp

from utils import dims, stack_states
from equilibrium import split_z
from probabilities import ccps_raw
from transitions import physical_matrix_raw, transition_from_ccps, stationary_distribution

O_LABELS = np.array(["keep", "purge_vende", "purge_chatarra", "trade_vende", "trade_chatarra"])


def equilibrium_objects(z, eco):
    # Lista por tipo con todo lo que necesita la simulación, en numpy
    EVs, P = split_z(z, eco)
    return [type_objects(EVs[t], P, eco.type_model(t)) for t in range(eco.n_types)]


def type_objects(EV, P, m):
    J, A, n_act, n = dims(m)
    c = ccps_raw(EV, P, m)
    q = stationary_distribution(transition_from_ccps(c, m))
    scrap_x = stack_states(c.scrap, jnp.ones(J), jnp.zeros(()))
    out = dict(P=P, EV=EV, q=q, keep=c.keep, purge=c.purge, trade=c.trade, buy=c.buy,
               scrap_x=scrap_x, Q=physical_matrix_raw(m))
    return {k: np.asarray(v) for k, v in out.items()}


def decode(idx, g):
    # índice de X (o de H) -> (tipo, j, edad).  tipo: 0 activo/usado, 1 terminal/nuevo, 2 sin coche
    J, A, n_act, n = dims(g)
    idx = np.asarray(idx)
    kind = np.where(idx < n_act, 0, np.where(idx < n_act + J, 1, 2))
    j = np.where(kind == 0, idx // (A - 1), np.where(kind == 1, idx - n_act, -1))
    a = np.where(kind == 0, idx % (A - 1) + 1, np.where(kind == 1, 0, -1))
    return kind, j, a


def _sample_rows(cum, rows, u):
    return np.minimum((cum[rows] < u[:, None]).sum(axis=1), cum.shape[1] - 1)


def _sample_cat(p, size, rng):
    p = np.clip(p, 0.0, None)
    return rng.choice(len(p), size=size, p=p / p.sum())


def simulate_panel(eqos, g, f, N, K, seed):
    # eqos: lista por tipo (equilibrium_objects); f: fracciones
    rng = np.random.default_rng(seed)
    frames, start = [], 0
    for t, (eqo, ft) in enumerate(zip(eqos, f)):
        Nt = int(round(N * ft))
        d = _simulate_type(eqo, g, Nt, K, rng)
        d["id_hogar"] += start
        d.insert(1, "tipo", t)
        frames.append(d)
        start += Nt
    return pd.concat(frames, ignore_index=True)


def _simulate_type(eqo, g, N, K, rng):
    J, A, n_act, n = dims(g)
    cumQ = np.cumsum(eqo["Q"], axis=1)
    keep, purge, scrap_x = eqo["keep"], eqo["purge"], eqo["scrap_x"]

    x = _sample_cat(eqo["q"], N, rng)
    frames = []
    for k in range(K):
        u = rng.random(N)
        dec = np.where(u < keep[x], 0, np.where(u < keep[x] + purge[x], 1, 2))
        sc = (rng.random(N) < scrap_x[x]).astype(np.int64)        # 1 en terminal, 0 sin coche
        o = np.where(dec == 0, 0, np.where(dec == 1, 1 + sc, 3 + sc))
        h = np.where(dec == 0, x, n - 1)
        is_trade = dec == 2
        h[is_trade] = _sample_cat(eqo["buy"], int(is_trade.sum()), rng)
        x_next = _sample_rows(cumQ, h, rng.random(N))

        kx, jx, ax = decode(x, g)
        kh, jh, dh = decode(h, g)
        frames.append(pd.DataFrame({
            "id_hogar": np.arange(N), "k": k, "x": x, "o": o, "h": h,
            "estado": np.array(["activo", "terminal", "sin_coche"])[kx], "j": jx, "a": ax,
            "decision": O_LABELS[o],
            "chatarreo_endogeno": ((kx == 0) & (dec > 0) & (sc == 1)).astype(np.int8),
            "tipo_h": np.array(["usado", "nuevo", "sin_coche"])[kh], "j_h": jh, "d_h": dh,
        }))
        x = x_next

    df = pd.concat(frames, ignore_index=True)
    # Accidente: el coche de la tenencia anterior llegó a terminal antes de la edad máxima
    df = df.sort_values(["id_hogar", "k"], ignore_index=True)
    prev_d = df.groupby("id_hogar")["d_h"].shift(1)
    prev_t = df.groupby("id_hogar")["tipo_h"].shift(1)
    df["accidente"] = ((df["estado"] == "terminal") & (prev_t != "sin_coche")
                       & (prev_d < A - 1)).astype(np.int8)
    return df


def to_cells(df):
    # Transiciones h_{k-1} -> (x_k, o_k, h_k) agregadas a celdas con conteos (ec. 40).
    # Se condiciona en la tenencia anterior, así que el periodo k = 0 solo aporta h_0.
    d = df.copy()
    d["h_prev"] = d.groupby("id_hogar")["h"].shift(1)
    d = d.dropna(subset=["h_prev"])
    d["h_prev"] = d["h_prev"].astype(np.int64)
    cells = d.groupby(["tipo", "h_prev", "x", "o", "h"]).size().rename("cnt").reset_index()
    return cells
