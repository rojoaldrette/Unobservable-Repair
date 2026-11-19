# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/gillingham/gen_dataset.py
# Goal:           Panel simulado de hogares en el equilibrio estacionario y celdas
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           07/10/2026
#
# _____________________________________________________________________________
'''
Como el registro danés del paper: N hogares (round(N f_t) de cada tipo, tipo observado)
seguidos K años, con el estado inicial sacado de la distribución estacionaria q_t
(sec. 5.1: los datos vienen del estado estacionario).  Cada año:

    x (inicio)  ->  resultado o (probabilidades.OUTCOMES) y tenencia h
                ->  x' ~ Q[h]  (envejece o accidente)

Columnas del panel:
    id, tipo, k, x, o, h           lo que usa la verosimilitud (índices de X y H)
    accidente                      verdad no observada: el coche de h se accidentó al final
                                   del año k (x_{k+1} terminal antes de la edad máxima)
    chatarreo_endogeno             verdad: el hogar se deshizo de un coche activo y lo chatarreó

Celdas (ec. 40): transiciones h_{k-1} -> (x_k, o_k, h_k) con conteos.  El año k = 0 solo
aporta h_0 (se condiciona en la tenencia anterior, como en el apéndice D).
    "completa":  (tipo, h_prev, x, o, h)   se observa el accidente (x terminal)
    "parcial":   (tipo, h_prev, o, h)      la del paper: x no se observa cuando hay salida
'''

import numpy as np
import pandas as pd

from utils import dims


def _draw(p, size, rng):
    p = np.clip(p, 0, None)
    return rng.choice(len(p), size=size, p=p / p.sum())


def _draw_rows(cum, rows, rng):
    # Una extracción por fila de una matriz de probabilidades acumuladas
    u = rng.random(len(rows))
    return np.minimum((cum[rows] < u[:, None]).sum(axis=1), cum.shape[1] - 1)


def simulate_type(o, cfg, N, K, rng):
    # o: objetos de equilibrio de un tipo (equilibrio.equilibrium_objects)
    J, A, n_act, n = dims(cfg)
    cumQ = np.cumsum(o["Q"], axis=1)
    keep, purge, scrap = o["keep"], o["purge"], o["scrap"]
    last_used = (A - 1) * np.arange(J) + A - 2                  # used(j, A-1): terminal seguro
    is_term = lambda v: (v >= n_act) & (v < n_act + J)

    x = _draw(o["q"], N, rng)
    rows = []
    for k in range(K):
        u = rng.random(N)
        dec = np.where(u < keep[x], 0, np.where(u < keep[x] + purge[x], 1, 2))   # keep/purge/trade
        sc = (rng.random(N) < scrap[x]).astype(int)              # 1 en terminal, 0 sin coche
        out = np.where(dec == 0, 0, np.where(dec == 1, 1 + sc, 3 + sc))
        h = np.where(dec == 0, x, n - 1)
        tr = dec == 2
        h[tr] = _draw(o["buy"], int(tr.sum()), rng)
        x_next = _draw_rows(cumQ, h, rng)
        rows.append(pd.DataFrame(dict(
            id=np.arange(N), k=k, x=x, o=out, h=h,
            accidente=(is_term(x_next) & (h < n - 1) & ~np.isin(h, last_used)).astype(np.int8),
            chatarreo_endogeno=((x < n_act) & (dec > 0) & (sc == 1)).astype(np.int8))))
        x = x_next
    return pd.concat(rows, ignore_index=True)


def simulate_panel(objs, cfg, N, K, seed):
    rng = np.random.default_rng(seed)
    out, start = [], 0
    for t, (o, f) in enumerate(zip(objs, cfg.f)):
        Nt = int(round(N * f))
        d = simulate_type(o, cfg, Nt, K, rng)
        d["id"] += start
        d.insert(1, "tipo", t)
        out.append(d)
        start += Nt
    return pd.concat(out, ignore_index=True).sort_values(["id", "k"], ignore_index=True)


def to_cells(df, info):
    d = df.assign(h_prev=df.groupby("id")["h"].shift(1)).dropna(subset=["h_prev"])
    d["h_prev"] = d["h_prev"].astype(np.int64)
    keys = ["tipo", "h_prev", "x", "o", "h"] if info == "completa" else ["tipo", "h_prev", "o", "h"]
    return d.groupby(keys).size().rename("cnt").reset_index()
