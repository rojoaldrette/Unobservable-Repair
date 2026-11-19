# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/final/modelo_tesis/inferencia.py
# Goal:           Inferencia de la reparación no observada, observación por observación
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           18/11/2026
#
# _____________________________________________________________________________
'''
Para cada tenencia reparable h = used(j, a, w), a = 1..A-2, del panel (la de un año k) se
mira lo que pasó al año siguiente (x_{k+1}, o_{k+1}) y se calcula, con un θ y sus equilibrios:

    prior    = Pr(r = 1 | h)                          (la CCP; antes de ver el año siguiente)
    post_d2  = Pr(r = 1 | h, x)                       (lo que ve el diseño 2: w' si sobrevive)
             = p F_1(w' | h) / [p F_1(w' | h) + (1 - p) F_0(w' | h)]       si sobrevive
             = p                                                           si accidente
    post_d3  = Pr(r = 1 | h, o)  si el coche salió (o = chatarrea): el diseño 3 no ve x
             = p L_1 / [p L_1 + (1 - p) L_0],
               L_r = s C[term, o] + (1 - s) sum_w' F_r(w' | h) C[act(j, a+1, w'), o]
             = post_d2           si no salió

C y la compra se cancelan en el cociente cuando x se observa.  Es la misma mezcla de la
verosimilitud (ll_estim.cell_logp), invertida con la regla de Bayes.

Se agrega por celda (régimen, tipo, h, x, o) con n = observaciones y n_r1 = cuántas
repararon de verdad: así el archivo es mucho más chico que uno por hogar y no se pierde nada
(todas las observaciones de una celda tienen las mismas probabilidades).
'''

import numpy as np
import pandas as pd

from params import regime_theta
from utils import dims
from bellman import breakdown_prob, w_transition
from gen_dataset import decode
from ll_estim import regime_objects


def celdas_reparables(df, cfg):
    # (regimen, tipo, h_prev, x, o) con conteos, solo tenencias reparables
    d = df.assign(h_prev=df.groupby("id")["h"].shift(1), r_prev=df.groupby("id")["r"].shift(1))
    d = d.dropna(subset=["h_prev"]).astype({"h_prev": np.int64, "r_prev": np.int64})
    d = d[d["r_prev"] >= 0]                                 # r = -1: no se podía reparar
    g = d.groupby(["regimen", "tipo", "h_prev", "x", "o"])["r_prev"]
    return pd.DataFrame(dict(n=g.size(), n_r1=g.sum())).reset_index()


def posteriores(cells, zs, th, cfg):
    # prior, post_d2 y post_d3 por celda, con θ = th y equilibrios zs (uno por régimen)
    J, A, W, n_act, n = dims(cfg)
    objs = [regime_objects(zs[t], regime_theta(th, cfg, t), cfg) for t in range(len(cfg.zetas))]
    C = np.stack([np.asarray(o[0]) for o in objs])         # (R, T, n, 5)
    rep = np.stack([np.asarray(o[2]) for o in objs])       # (R, T, J, A-1, W)
    s = np.asarray(breakdown_prob(th, cfg))                 # (J, A, W)
    F = np.asarray(w_transition(th, cfg))                   # (2, J, W, W)

    t, tau = cells["regimen"].to_numpy(), cells["tipo"].to_numpy()
    x, o = cells["x"].to_numpy(), cells["o"].to_numpy()
    j, d, w = decode(cells["h_prev"].to_numpy(), cfg, "H")
    x_term = (x >= n_act) & (x < n_act + J)
    w_next = np.where(x_term, 0, decode(np.minimum(x, n_act - 1), cfg, "X")[2])

    p = rep[t, tau, j, d - 1, w]
    sh = s[j, d, w]
    f0, f1 = F[0, j, w, w_next], F[1, j, w, w_next]
    post2 = np.where(x_term, p, p * f1 / (p * f1 + (1 - p) * f0))

    salida = np.isin(o, (2, 4))
    Ct = C[t, tau, n_act + j, o]
    idx = (j * (A - 1) * W + d * W)[:, None] + np.arange(W)[None, :]      # act(j, a+1, ·)
    Cn = C[t[:, None], tau[:, None], idx, o[:, None]]                      # (celdas, W)
    L = [sh * Ct + (1 - sh) * (F[r, j, w, :] * Cn).sum(-1) for r in (0, 1)]
    post3 = np.where(salida, p * L[1] / (p * L[1] + (1 - p) * L[0]), post2)

    return pd.DataFrame(dict(
        regimen=t, tipo=tau, j=j, a=d, w=w, accidente=x_term.astype(int),
        w_next=np.where(x_term, -1, w_next), o=o, salida=salida.astype(int),
        n=cells["n"].to_numpy(), n_r1=cells["n_r1"].to_numpy(),
        prior=p, post_d2=post2, post_d3=post3))


def auc(p, pos, neg):
    # Área bajo la curva ROC con pesos (celdas): Pr(p de un positivo > p de un negativo),
    # empates cuentan 1/2
    g = pd.DataFrame(dict(p=p, pos=pos, neg=neg)).groupby("p", sort=True)[["pos", "neg"]].sum()
    neg_antes = g["neg"].cumsum() - g["neg"]
    P, N = g["pos"].sum(), g["neg"].sum()
    return float((g["pos"] * (neg_antes + 0.5 * g["neg"])).sum() / (P * N)) if P * N > 0 else np.nan


def resumen(d):
    # Una fila por (rep, estimador, info): tasa real, media, Brier y AUC (todas las réplicas)
    out = []
    for (rep, est), g in d.groupby(["rep", "estimador"], sort=False):
        n, pos = g["n"].to_numpy(), g["n_r1"].to_numpy()
        neg = n - pos
        for info in ("prior", "post_d2", "post_d3"):
            p = g[info].to_numpy()
            out.append(dict(rep=rep, estimador=est, info=info, n=int(n.sum()), tasa_real=pos.sum() / n.sum(),
                            media=(n * p).sum() / n.sum(),
                            brier=(pos * (1 - p) ** 2 + neg * p ** 2).sum() / n.sum(),
                            auc=auc(p, pos, neg)))
    return pd.DataFrame(out)
