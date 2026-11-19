# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/modelo_tesis/probabilidades.py
# Goal:           CCPs, transición física y distribución estacionaria (un tipo)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
Sin matrices n x n: la física se escribe con núcleos por edad.

    K_d[j, w, w'] = (1 - s(j, d, w)) [ (1 - p(j, d, w)) F_0 + p(j, d, w) F_1 ](w' | j, w)

es la masa que pasa de la tenencia used(j, d, w) a act(j, d+1, w') (la que muere va a
term(j)).  Un nuevo pasa a act(j, 1, w') con (1 - s(j, 0, 0)) F_0(w' | j, 0).

Distribución estacionaria (q = q M), por recursión en la edad.  Todo coche activo viene
de una compra y las compras se reparten según buy(h), que no depende de x.  Con una masa
compradora de 1:

    act(j, 1)    = buy_new(j) (1 - s0) F_0(· | 0)
    hold(j, d)   = act(j, d) keep(j, d) + buy_used(j, d)
    act(j, d+1)  = hold(j, d) @ K_d
    term(j)      = sum_d hold(j, d) · s(j, d) + hold(j, A-1) + buy_new(j) s0
    none         = (lo que entra por purge) / Pr(trade | none)

y se normaliza.  Solo suma productos no negativos: precisión relativa de máquina aun en
celdas de 1e-15.  `advance` aplica un periodo completo (comercio y física) para comprobarlo.

Resultados observables al inicio de cada año (OUTCOMES, como en niu/gillingham):
    0 keep | 1 purge vende | 2 purge chatarrea | 3 trade vende | 4 trade chatarrea
'''

from typing import NamedTuple

import jax
import jax.numpy as jnp

from utils import dims, split_states, stack_states, choice_probs
from bellman import choice_values, breakdown_prob, w_transition

OUTCOMES = ("keep", "purge_vende", "purge_chatarra", "trade_vende", "trade_chatarra")


class CCP(NamedTuple):
    keep: jnp.ndarray       # (n,) sobre X
    purge: jnp.ndarray      # (n,) sobre X
    trade: jnp.ndarray      # (n,) sobre X
    scrap: jnp.ndarray      # (n,) sobre X: Pr(chatarrear | se deshace); 1 en term, 0 en none
    buy: jnp.ndarray        # (n,) sobre H: Pr(h | trade); 0 en none
    lp_used: jnp.ndarray    # (J, A-1, W): log Pr(used(j, d, w) | trade), exacto
    repair: jnp.ndarray     # (J, A-1, W): Pr(reparar | used(j, d, w))


def ccps(EV, P, th, cfg):
    J, A, W, n_act, n = dims(cfg)
    v = choice_values(EV, P, th, cfg)
    p_act = choice_probs([v["keep"], v["purge_act"], v["trade_act"]], cfg.sigma)
    p_term = choice_probs([v["purge_term"], v["trade_term"]], cfg.sigma)
    p_none = choice_probs([v["stay_none"], v["trade_none"]], cfg.sigma)
    scrap = choice_probs([v["scrap"], v["sell"]], th["sigma_sell"])[0]
    zJ, z0 = jnp.zeros(J), jnp.zeros(())
    buy = jnp.concatenate([jnp.exp(jnp.ravel(v["lp_used"])), jnp.exp(v["lp_new"]), jnp.zeros(1)])
    return CCP(keep=stack_states(p_act[0], zJ, z0),
               purge=stack_states(p_act[1], p_term[0], p_none[0]),
               trade=stack_states(p_act[2], p_term[1], p_none[1]),
               scrap=stack_states(scrap, jnp.ones(J), z0),
               buy=buy, lp_used=v["lp_used"], repair=v["p_rep"])


def kernels(c, th, cfg):
    # K (A-2, J, W, W) para d = 1..A-2; masa a act(j, 1, ·) por cada nuevo comprado (J, W);
    # s (J, A, W)
    J, A, W, n_act, n = dims(cfg)
    s = breakdown_prob(th, cfg)
    F = w_transition(th, cfg)
    p = c.repair[:, :A - 2, :, None]                                        # (J, A-2, W, 1)
    K = (1 - s[:, 1:A - 1, :, None]) * ((1 - p) * F[0][:, None] + p * F[1][:, None])
    k_new = (1 - s[:, 0, cfg.w0])[:, None] * F[0, :, cfg.w0, :]
    return jnp.moveaxis(K, 1, 0), k_new, s


def stationary(c, th, cfg):
    J, A, W, n_act, n = dims(cfg)
    K, k_new, s = kernels(c, th, cfg)
    keep, _, _ = split_states(c.keep, cfg)
    purge_a, purge_t, _ = split_states(c.purge, cfg)
    _, _, trade_none = split_states(c.trade, cfg)
    b_used = c.buy[:n_act].reshape(J, A - 1, W)
    b_new = c.buy[n_act:n_act + J]

    def step(act_d, xs):
        keep_d, b_d, K_d = xs
        hold = act_d * keep_d + b_d
        return jnp.einsum("jw,jwv->jv", hold, K_d), (act_d, hold)

    xs = (jnp.moveaxis(keep[:, :A - 2], 1, 0), jnp.moveaxis(b_used[:, :A - 2], 1, 0), K)
    last, (acts, holds) = jax.lax.scan(step, b_new[:, None] * k_new, xs)
    act = jnp.concatenate([jnp.moveaxis(acts, 0, 1), last[:, None]], axis=1)      # (J, A-1, W)
    hold_last = last * keep[:, A - 2] + b_used[:, A - 2]
    term = (jnp.einsum("djw,jdw->j", holds, s[:, 1:A - 1]) + hold_last.sum(-1)
            + b_new * s[:, 0, cfg.w0])
    none = ((act * purge_a).sum() + (term * purge_t).sum()) / trade_none
    q = stack_states(act, term, none)
    return q / q.sum()


def advance(q, c, th, cfg):
    # Un periodo completo desde q (inicio de año) -> q' (inicio del año siguiente)
    J, A, W, n_act, n = dims(cfg)
    K, k_new, s = kernels(c, th, cfg)
    act, term, none = split_states(q, cfg)
    keep, _, _ = split_states(c.keep, cfg)
    m = q @ c.trade
    hold = act * keep + m * c.buy[:n_act].reshape(J, A - 1, W)
    h_new = m * c.buy[n_act:n_act + J]
    act2 = jnp.concatenate([(h_new[:, None] * k_new)[:, None],
                            jnp.einsum("jdw,djwv->jdv", hold[:, :A - 2], K)], axis=1)
    term2 = (hold[:, :A - 2] * s[:, 1:A - 1]).sum((1, 2)) + hold[:, A - 2].sum(-1) + h_new * s[:, 0, cfg.w0]
    return stack_states(act2, term2, q @ c.purge)


def outcome_probs(c):
    # C (n, 5) sobre X
    return jnp.stack([c.keep, c.purge * (1 - c.scrap), c.purge * c.scrap,
                      c.trade * (1 - c.scrap), c.trade * c.scrap], axis=1)
