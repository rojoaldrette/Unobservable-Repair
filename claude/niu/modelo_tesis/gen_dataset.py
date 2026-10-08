# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/modelo_tesis/gen_dataset.py
# Goal:           Panel simulado de hogares desde los equilibrios de cada régimen
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
Por régimen t y tipo: round(N f_tipo) hogares (N por régimen), K años, estado inicial de
la distribución estacionaria del régimen.  Cada año:

    x = (j, a, w) o terminal o sin coche
      -> resultado o (probabilidades.OUTCOMES) y tenencia h
      -> etapa 2: r ~ Bernoulli(Pr(reparar | h)) si h es un usado de edad 1..A-2 (si no, r = -1)
      -> se maneja h: se descompone con prob s(h) (accidente) o, si sobrevive,
         w' ~ F_r(· | j, w) y edad d+1;  un usado de edad A-1 llega a la terminal

Columnas (una fila por hogar y año):
    regimen, tipo, id, k            hogar y año
    x, o, h                         índices de X y H y resultado (lo que usa la verosimilitud)
    j, a, w, j_h, d_h, w_h          decodificados (a = 0 terminal, -1 sin coche; d_h = 0 nuevo)
    r                               reparó h este año (1/0; -1 si no aplica)
    accidente                       h se descompuso este año (antes de la edad A-1)
    chatarreo_endogeno              se deshizo de un activo y lo chatarreó

Qué ve cada diseño (modelo_fin.md, sec. 6) se decide al armar las celdas en la estimación:
diseño 1 usa r y el motivo de salida; 2 borra r; 3 borra r y el motivo (accidente vs.
chatarreo voluntario).
'''

import numpy as np
import pandas as pd

from utils import dims


def decode(idx, cfg, space="X"):
    # índice -> (j, edad, w_idx).  X: edad a (0 = terminal, -1 = sin coche); H: d (0 = nuevo)
    J, A, W, n_act, n = dims(cfg)
    idx = np.asarray(idx)
    act = idx < n_act
    tn = (idx >= n_act) & (idx < n_act + J)
    j = np.where(act, idx // ((A - 1) * W), np.where(tn, idx - n_act, -1))
    age = np.where(act, (idx // W) % (A - 1) + 1, np.where(tn, 0, -1))
    w = np.where(act, idx % W, np.where(tn & (space == "H"), cfg.w0, -1))
    return j, age, w


def _draw(p, size, rng):
    p = np.clip(p, 0, None)
    return rng.choice(len(p), size=size, p=p / p.sum())


def _draw_rows(cum, rng):
    # una extracción por fila de probabilidades acumuladas (m, W)
    u = rng.random(cum.shape[0])
    return np.minimum((cum < u[:, None]).sum(axis=1), cum.shape[1] - 1)


def simulate_type(o, cfg, N, K, rng):
    J, A, W, n_act, n = dims(cfg)
    keep, purge, scrap, buy = o["keep"], o["purge"], o["scrap"], o["buy"]
    rep = o["repair"].ravel()
    cumF = np.cumsum(o["F"], axis=-1)                                     # (2, J, W, W)
    s = o["s"]                                                            # (J, A, W)

    x = _draw(o["q"], N, rng)
    rows = []
    for k in range(K):
        u = rng.random(N)
        dec = np.where(u < keep[x], 0, np.where(u < keep[x] + purge[x], 1, 2))
        sc = (rng.random(N) < scrap[x]).astype(int)
        out = np.where(dec == 0, 0, np.where(dec == 1, 1 + sc, 3 + sc))
        h = np.where(dec == 0, x, n - 1)
        tr = dec == 2
        h[tr] = _draw(buy, int(tr.sum()), rng)

        # Etapa 2 y física
        jh, dh, wh = decode(h, cfg, "H")
        used, new = h < n_act, (h >= n_act) & (h < n - 1)
        can = used & (dh <= A - 2)
        r = np.full(N, -1)
        r[can] = (rng.random(int(can.sum())) < rep[h[can]]).astype(int)
        s_h = np.zeros(N)
        s_h[used] = s[jh[used], dh[used], wh[used]]
        s_h[new] = s[jh[new], 0, cfg.w0]
        last = used & (dh == A - 1)
        die = (used | new) & ~last & (rng.random(N) < s_h)
        surv = (used | new) & ~last & ~die
        w_next = np.full(N, -1)
        rr = np.maximum(r, 0)
        w_next[surv] = _draw_rows(cumF[rr[surv], jh[surv], wh[surv]], rng)
        a_next = dh + 1                                                   # nuevo (d = 0) -> 1
        x_next = np.full(N, n - 1)
        x_next[die | last] = n_act + jh[die | last]
        x_next[surv] = (jh[surv] * (A - 1) + a_next[surv] - 1) * W + w_next[surv]

        jx, ax, wx = decode(x, cfg, "X")
        rows.append(pd.DataFrame(dict(
            id=np.arange(N), k=k, x=x, o=out, h=h, j=jx, a=ax, w=wx, j_h=jh, d_h=dh, w_h=wh,
            r=r, accidente=die.astype(np.int8),
            chatarreo_endogeno=((x < n_act) & (dec > 0) & (sc == 1)).astype(np.int8))))
        x = x_next
    return pd.concat(rows, ignore_index=True)


def simulate_panel(objs_by_regime, cfg, N, K, seed):
    # objs_by_regime[t]: equilibrium_objects del régimen t.  N hogares por régimen.
    rng = np.random.default_rng(seed)
    out, start = [], 0
    for t, objs in enumerate(objs_by_regime):
        for tau, (o, f) in enumerate(zip(objs, cfg.f)):
            Nt = int(round(N * f))
            d = simulate_type(o, cfg, Nt, K, rng)
            d["id"] += start
            d.insert(0, "tipo", tau)
            d.insert(0, "regimen", t)
            out.append(d)
            start += Nt
    return pd.concat(out, ignore_index=True).sort_values(["id", "k"], ignore_index=True)
