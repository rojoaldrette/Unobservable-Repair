# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/modelo_tesis/ll_estim.py
# Goal:           Verosimilitud de los 3 diseños, gradiente implícito y matriz BHHH
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
Núcleo de la estimación (el optimizador y el Monte Carlo van aparte).  modelo_fin.md, sec. 6.

Observación = un hogar en una transición h_{k-1} -> (x_k, o_k, h_k), con r_{k-1} (si
reparó h_{k-1} el año anterior).  Se agrega a celdas con conteos.  Con
C[x, o] = Pr(o | x), buy(h), p(h) = Pr(reparar | h), s(h) y F_r(w' | h):

    Pr(x | h, r) = s(h)                         si x = term (accidente o fin de vida)
                 = (1 - s(h)) F_r(w' | h)        si x = act(j, d+1, w')
    (h usado de edad A-1: x = term con prob 1;  h nuevo: r = 0;  h sin coche: x = none)

    diseño 1 (r y x observados):   Pr(r | h) Pr(x | h, r) · C[x, o] · buy(h_k)^{trade}
    diseño 2 (x observado):        [sum_r Pr(r | h) Pr(x | h, r)] · C[x, o] · buy^{trade}
    diseño 3 (si el coche de h salió del parque (o = chatarrea) no se observa x, porque no
              se sabe si fue accidente o chatarreo voluntario):
        salidas:  sum_r Pr(r | h) [ s C[term, o] + (1 - s) sum_w' F_r(w' | h) C[act(j, d+1, w'), o] ] · buy^{trade}
        resto:    como el diseño 2

Precios no observados: P(θ) del equilibrio de cada régimen.  Gradiente por la función
implícita, régimen por régimen: dz_t/dx = -F_z^{-1} F_x, con la LU de F_z que ya calculó el
solver (equilibrio.factor) más pasos de refinamiento (la LU puede ser de un θ cercano).
Scores por celda -> gradiente y BHHH.

Parámetros libres x <-> θ: mu, sigma_eta, sigma_rep, kappa = exp(x);
sigma_sell = sigma_trade sigmoid(x); el resto = x.  Fijos: sigma, sigma_trade, sigma_w,
beta, p_new, p_scrap, u2 y R (dato).
'''

from functools import partial

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_solve

from params import TYPE_KEYS, regime_theta
from utils import dims, split_states, type_axes
from bellman import breakdown_prob, w_transition
from probabilidades import ccps, outcome_probs
from equilibrio import split_z, residual, per_type
from gen_dataset import decode

FREE = ("mu", "u0", "u1", "tc_buy", "tc_buy_nocar", "tc_sell", "tc_sell_inspect", "sigma_sell",
        "acc_int", "acc_age", "u_w", "delta", "kappa", "sigma_eta", "sigma_rep")
LOG = ("mu", "kappa", "sigma_eta", "sigma_rep")
DESIGNS = (1, 2, 3)


# Vector libre ______________________________________________________________

def free_spec(th, fix=()):
    return tuple((k, tuple(np.shape(th[k]))) for k in FREE if k not in fix)


def _fwd(k, v, cfg):
    if k in LOG:
        return jnp.exp(v)
    if k == "sigma_sell":
        return cfg.sigma_trade * jax.nn.sigmoid(v)
    return v


def _inv(k, v, cfg):
    if k in LOG:
        return jnp.log(v)
    if k == "sigma_sell":
        r = v / cfg.sigma_trade
        return jnp.log(r / (1 - r))
    return v


def pack(th, spec, cfg):
    return jnp.concatenate([jnp.ravel(_inv(k, th[k], cfg)) for k, _ in spec])


def unpack(x, spec, th_fixed, cfg):
    th, i = dict(th_fixed), 0
    for k, shape in spec:
        size = int(np.prod(shape))
        th[k] = _fwd(k, x[i:i + size].reshape(shape), cfg)
        i += size
    return th


def natural(x, spec, th_fixed, cfg):
    th = unpack(x, spec, th_fixed, cfg)
    return jnp.concatenate([jnp.ravel(th[k]) for k, _ in spec])


def labels(spec):
    out = []
    for k, shape in spec:
        for idx in np.ndindex(*shape):
            parts = [k]
            if k in TYPE_KEYS:
                parts.append(f"t{idx[0]}")
                idx = idx[1:]
            parts += [f"j{i}" for i in idx]
            out.append("_".join(parts))
    return out


# Celdas ______________________________________________________________

def to_cells(df, cfg, design):
    # Panel de gen_dataset -> celdas con conteos.  Se condiciona en h_{k-1}: el año k = 0
    # de cada hogar solo aporta h_0.
    J, A, W, n_act, n = dims(cfg)
    d = df.assign(h_prev=df.groupby("id")["h"].shift(1), r_prev=df.groupby("id")["r"].shift(1))
    d = d.dropna(subset=["h_prev"]).astype({"h_prev": np.int64, "r_prev": np.int64})
    if design != 1:
        d["r_prev"] = -1
    if design == 3:
        salida = (d["h_prev"] < n - 1) & d["o"].isin([2, 4])
        d.loc[salida, "x"] = -1
    keys = ["regimen", "tipo", "h_prev", "r_prev", "x", "o", "h"]
    return d.groupby(keys).size().rename("cnt").reset_index()


def cell_data(cells, cfg):
    J, A, W, n_act, n = dims(cfg)
    hp = cells["h_prev"].to_numpy()
    x = cells["x"].to_numpy()
    o = cells["o"].to_numpy()
    jh, dh, wh = decode(hp, cfg, "H")
    kind = np.where(hp < n_act, 0, np.where(hp < n - 1, 1, 2))              # usado / nuevo / sin coche
    _, ax, wx = decode(np.maximum(x, 0), cfg, "X")
    d = dict(
        t=cells["regimen"].to_numpy(), tau=cells["tipo"].to_numpy(), o=o, h=cells["h"].to_numpy(),
        x=np.maximum(x, 0), x_obs=x >= 0, x_term=(x >= n_act) & (x < n - 1),
        w_next=np.maximum(wx, 0),
        r=cells["r_prev"].to_numpy(), cnt=cells["cnt"].to_numpy().astype(float),
        car=kind < 2, used=kind == 0,
        last=(kind == 0) & (dh == A - 1),
        can_rep=(kind == 0) & (dh <= A - 2),
        j=np.maximum(jh, 0), d=np.maximum(dh, 0), w=np.where(kind == 0, wh, cfg.w0),
        trade=(o >= 3).astype(float),
    )
    return {k: jnp.asarray(v) for k, v in d.items()}


# Probabilidades por celda ______________________________________________________________

def _log(p):
    return jnp.log(jnp.clip(p, 1e-300, None))


def _type_objs(EV, P, th, cfg):
    c = ccps(EV, P, th, cfg)
    return outcome_probs(c), c.buy, c.repair


def regime_objects(z, th_t, cfg):
    # (C (T, n, 5), buy (T, n), Pr(reparar) (T, J, A-1, W)) de un régimen
    EVs, P = split_z(z, cfg)
    return per_type(_type_objs, EVs, P, th_t, cfg)


def cell_logp(zs, th, cfg, data, design):
    J, A, W, n_act, n = dims(cfg)
    R = len(cfg.zetas)
    objs = [regime_objects(zs[t], regime_theta(th, cfg, t), cfg) for t in range(R)]
    C, buy, rep = (jnp.stack(v) for v in zip(*objs))                    # (R, T, ...)
    s = breakdown_prob(th, cfg)                                          # (J, A, W)
    F = w_transition(th, cfg)                                            # (2, J, W, W)
    t, tau, j, d, w = data["t"], data["tau"], data["j"], data["d"], data["w"]

    s_h = jnp.where(data["last"], 1.0, s[j, d, w])                       # nuevo: d = 0, w = w0
    p1 = jnp.where(data["can_rep"], rep[t, tau, j, jnp.maximum(d - 1, 0), w], 0.0)
    b = jnp.where(data["trade"] > 0, buy[t, tau, data["h"]], 1.0)       # (no buy ** trade: NaN en la derivada)
    choice = C[t, tau, data["x"], data["o"]] * b

    # x observado
    surv = [(1 - s_h) * F[r, j, w, data["w_next"]] for r in (0, 1)]
    px = [jnp.where(data["x_term"], s_h, surv[r]) for r in (0, 1)]
    if design == 1:
        r1 = data["r"] == 1
        pr = jnp.where(data["can_rep"], jnp.where(r1, p1, 1 - p1), 1.0)
        trans = pr * jnp.where(r1, px[1], px[0])
    else:
        trans = (1 - p1) * px[0] + p1 * px[1]
    lp = _log(jnp.where(data["car"], trans, 1.0)) + _log(choice)

    if design == 3:
        # Salidas: x no se observa.  C de act(j, d+1, ·) para el resultado o de la celda.
        Cact = C[:, :, :n_act].reshape(R, -1, J, A - 1, W, 5)
        nxt = jnp.minimum(d, A - 2)                                      # índice de la edad d+1
        Cn = Cact[t, tau, j, nxt][jnp.arange(t.shape[0]), :, data["o"]]  # (celdas, W)
        Ct = C[t, tau, n_act + j, data["o"]]
        mix = sum(pr_r * (s_h * Ct + (1 - s_h) * (F[r, j, w] * Cn).sum(-1))
                  for r, pr_r in ((0, 1 - p1), (1, p1)))
        lp_exit = _log(mix * b)
        lp = jnp.where(data["x_obs"], lp, lp_exit)
    return lp


# Gradiente implícito, scores y BHHH ______________________________________________________________

@partial(jax.jit, static_argnames=("cfg", "spec", "n_refine"))
def dz_dx(zs, lus, x, th_fixed, cfg, spec, n_refine=4):
    # dz_t/dx = -F_z^{-1} F_x por régimen, con la LU dada (de este θ o de uno cercano) y
    # n_refine pasos de refinamiento con productos F_z v exactos.
    out = []
    for t in range(len(cfg.zetas)):
        th_of = lambda x_: regime_theta(unpack(x_, spec, th_fixed, cfg), cfg, t)
        Fx = jax.jacfwd(lambda x_: residual(zs[t], th_of(x_), cfg))(x)
        _, jvp = jax.linearize(lambda z_: residual(z_, th_of(x), cfg), zs[t])
        X = -lu_solve(lus[t], Fx)
        for _ in range(n_refine):
            res = jax.vmap(jvp, in_axes=1, out_axes=1)(X) + Fx
            X = X - lu_solve(lus[t], res)
        out.append(X)
    return out


@partial(jax.jit, static_argnames=("cfg", "spec", "design"))
def score_parts(zs, Dz, x, th_fixed, data, cfg, spec, design):
    # LL, gradiente y matriz BHHH en x, dados los equilibrios zs(x) y sus derivadas Dz
    th_of = lambda x_: unpack(x_, spec, th_fixed, cfg)
    lp = cell_logp(zs, th_of(x), cfg, data, design)
    S = jax.jacfwd(lambda x_: cell_logp([z + D @ (x_ - x) for z, D in zip(zs, Dz)],
                                        th_of(x_), cfg, data, design))(x)
    cnt = data["cnt"]
    return cnt @ lp, cnt @ S, S.T @ (cnt[:, None] * S)


@partial(jax.jit, static_argnames=("cfg", "spec", "design"))
def loglik(zs, x, th_fixed, data, cfg, spec, design):
    return data["cnt"] @ cell_logp(zs, unpack(x, spec, th_fixed, cfg), cfg, data, design)
