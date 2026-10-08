# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/modelo_tesis/bellman.py
# Goal:           Primitivas, etapa de reparar, valores por elección y operador de Bellman
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
Un tipo de hogar (`th` rebanado por tipo) y un régimen (th["R"] ya es R_t).

Dentro del año, para quien empieza en x = (j, a, w):

    etapa 1 (sigma):       purge | keep | trade
        trade (sigma_trade):   nuevos j  y  nidos (j, d) de usados
            nido (j, d) (sigma_w):   qué w comprar
        al deshacerse de un activo (sigma_sell): vender (mu P - Ts) o chatarrear (mu p_scrap)
    -> tenencia h = (j, d, w)
    etapa 2 (sigma_rep):   reparar o no h (usados de edad 1..A-2)
    -> se maneja h: utilidad u(j, d, w); se descompone con prob s(j, d, w)  (opción 2: no
       depende de r); si sobrevive, w' = w + delta_j - acc_age_j - kappa r + eta, edad d+1

    s(j, d, w) = logit^-1(acc_int_j + acc_age_j d + w)
    u(j, d, w) = u0_j + u1_j d + u2_j d^2 + u_w w

La etapa 1 ve a h solo por W(h) (valor inclusivo de la etapa 2).  Un nuevo tiene w = 0 y
no se repara; la edad A-1 llega a la terminal de todos modos y tampoco se repara.
'''

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from utils import dims, split_states, stack_states, emax


# Primitivas ______________________________________________________________

def flow_utility(th, cfg):
    # u(j, d, w), d = 0..A-1: (J, A, W)
    d = jnp.arange(cfg.a_max)[None, :, None]
    w = jnp.asarray(cfg.w_grid)[None, None, :]
    return (th["u0"][:, None, None] + th["u1"][:, None, None] * d + th["u2"][:, None, None] * d ** 2
            + th["u_w"] * w)


def breakdown_prob(th, cfg):
    # s(j, d, w), d = 0..A-1: (J, A, W)
    d = jnp.arange(cfg.a_max)[None, :, None]
    w = jnp.asarray(cfg.w_grid)[None, None, :]
    return jax.nn.sigmoid(th["acc_int"][:, None, None] + th["acc_age"][:, None, None] * d + w)


def w_transition(th, cfg):
    # F[r, j, w, w'] (2, J, W, W): masa de N(m, sigma_eta^2) en el intervalo de w',
    # m = w + delta_j - acc_age_j - kappa r.  Los extremos absorben las colas.  Piso eps_F.
    g = jnp.asarray(cfg.w_grid)
    h = cfg.w_step
    # bordes extremos finitos (no ±inf): con inf, la derivada respecto a sigma_eta da 0·inf = NaN
    edges = jnp.concatenate([jnp.array([-1e3]), g[:-1] + h / 2, jnp.array([1e3])])
    drift = th["delta"] - th["acc_age"]                                       # (J,)
    m = (g[None, None, :] + drift[None, :, None]
         - th["kappa"] * jnp.arange(2.0)[:, None, None])                     # (2, J, W)
    cdf = norm.cdf((edges[None, None, None, :] - m[..., None]) / th["sigma_eta"])
    F = cdf[..., 1:] - cdf[..., :-1]
    return (1 - cfg.eps_F) * F + cfg.eps_F / cfg.W


def repair_allowed(cfg):
    # d = 1..A-1: se puede reparar en d <= A-2 (numpy: estático)
    return np.arange(1, cfg.a_max) <= cfg.a_max - 2


def inspection_years(cfg):
    a = np.arange(1, cfg.a_max)
    return (a % 2 == 0) & (a >= cfg.inspect_age_min)


def sell_cost(th, cfg):
    return jnp.where(jnp.asarray(inspection_years(cfg)), th["tc_sell_inspect"], th["tc_sell"])


def buy_costs(th, cfg):
    tb, tb_nc = th["tc_buy"], th["tc_buy_nocar"]
    return tb, (tb + tb_nc if cfg.nocar == "suma" else tb_nc)


# Etapa 2: reparar ______________________________________________________________

def continuation_values(EV, th, cfg):
    # cv[r] (2, J, A-1, W) = beta E[EV(x') | h = used(j, d, w), r],  d = 1..A-1
    # cv_new (J,) = beta E[EV(x') | h = new(j)]
    J, A, W, n_act, n = dims(cfg)
    act, term, _ = split_states(EV, cfg)
    F = w_transition(th, cfg)
    s = breakdown_prob(th, cfg)
    nxt = jnp.einsum("rjvw,jdw->rjdv", F, act[:, 1:, :])                     # d = 1..A-2 -> d+1
    sd = s[None, :, 1:A - 1, :]
    surv = (1 - sd) * nxt + sd * term[None, :, None, None]
    last = jnp.broadcast_to(term[None, :, None, None], (2, J, 1, W))
    cv = cfg.beta * jnp.concatenate([surv, last], axis=2)
    s0 = s[:, 0, cfg.w0]
    cv_new = cfg.beta * ((1 - s0) * (F[0, :, cfg.w0, :] * act[:, 0, :]).sum(-1) + s0 * term)
    return cv, cv_new


def repair_stage(EV, th, cfg):
    # Devuelve W(h) (J, A-1, W), Pr(reparar | h) (J, A-1, W) y cv_new (J,)
    cv, cv_new = continuation_values(EV, th, cfg)
    v0 = cv[0]
    v1 = -th["mu"] * th["R"][:, :, None] + cv[1]
    ok = jnp.asarray(repair_allowed(cfg))[None, :, None]
    Wv = jnp.where(ok, emax([v0, v1], th["sigma_rep"]), v0)
    p = jnp.where(ok, jax.nn.sigmoid((v1 - v0) / th["sigma_rep"]), 0.0)
    return Wv, p, cv_new


# Etapa 1 ______________________________________________________________

def buy_values(EV, P, th, cfg):
    # Valor de comprar cada h (sin Tb): usados (J, A-1, W), nuevos (J,); y W de la etapa 2
    u = flow_utility(th, cfg)
    Wv, p_rep, cv_new = repair_stage(EV, th, cfg)
    used = u[:, 1:, :] - th["mu"] * P + Wv
    new = u[:, 0, cfg.w0] - th["mu"] * th["p_new"] + cv_new
    return used, new, Wv, p_rep


def buy_logprobs(used, new, cfg):
    # log Pr(comprar h | trade): usados (J, A-1, W), nuevos (J,); y el valor inclusivo de trade
    I_jd = cfg.sigma_w * logsumexp(used / cfg.sigma_w, axis=-1)                # (J, A-1)
    iv = cfg.sigma_trade * logsumexp(jnp.concatenate([jnp.ravel(I_jd), new]) / cfg.sigma_trade)
    lp_used = (I_jd - iv)[..., None] / cfg.sigma_trade + (used - I_jd[..., None]) / cfg.sigma_w
    lp_new = (new - iv) / cfg.sigma_trade
    return lp_used, lp_new, iv


def choice_values(EV, P, th, cfg):
    u = flow_utility(th, cfg)
    used, new, Wv, p_rep = buy_values(EV, P, th, cfg)
    lp_used, lp_new, iv = buy_logprobs(used, new, cfg)
    _, _, none = split_states(EV, cfg)
    mu = th["mu"]
    tb, tb_none = buy_costs(th, cfg)
    sell = mu * P - sell_cost(th, cfg)[None, :, None]
    scrap = jnp.broadcast_to(mu * th["p_scrap"][:, None, None], P.shape)
    disposal = emax([scrap, sell], th["sigma_sell"])
    no_car = cfg.beta * none
    term_scrap = mu * th["p_scrap"]
    return dict(keep=u[:, 1:, :] + Wv,
                purge_act=disposal + no_car, trade_act=disposal - tb + iv,
                purge_term=term_scrap + no_car, trade_term=term_scrap - tb + iv,
                stay_none=no_car, trade_none=-tb_none + iv,
                sell=sell, scrap=scrap, lp_used=lp_used, lp_new=lp_new, p_rep=p_rep)


def T(EV, P, th, cfg):
    # Operador de Bellman sobre EV(x)
    v = choice_values(EV, P, th, cfg)
    return stack_states(emax([v["keep"], v["purge_act"], v["trade_act"]], cfg.sigma),
                        emax([v["purge_term"], v["trade_term"]], cfg.sigma),
                        emax([v["stay_none"], v["trade_none"]], cfg.sigma))


def successive_approx(EV, P, th, cfg, n_iter):
    return jax.lax.fori_loop(0, n_iter, lambda i, V: T(V, P, th, cfg), EV)
