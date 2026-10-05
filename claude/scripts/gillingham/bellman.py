# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/bellman.py
# Goal:           Primitivas, valores por elección, operador de Bellman y solver
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
Gillingham et al. (2019), sec. 3.3, ecs. (2)–(11), con el árbol de la figura 1:
    top (sigma):  purge | keep | trade
    trade (sigma_trade): todos los (j, d) y nuevos   [nidos de marca y edad colapsados]
    vender vs chatarrear el coche actual (sigma_sell): decisión estática (ec. 18),
    igual dentro de purge y de trade -> entra como un valor inclusivo `disposal`.
'''

from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxopt import FixedPointIteration

from utils import dims, split_states, stack_states


# Primitivas ______________________________________________________________

def flow_utility(g):
    # u(j, d) para d = 0..A-1.  shape (J, A)
    d = np.arange(g.a_max)
    even = ((d % 2 == 0) & (d >= g.inspect_age_min)).astype(float)
    d = jnp.asarray(d)[None, :]
    u0 = jnp.asarray(g.u0)[:, None]
    u1 = jnp.asarray(g.u1)[:, None]
    u2 = jnp.asarray(g.u2)[:, None]
    return u0 + u1 * d + u2 * d ** 2 + g.u_even * jnp.asarray(even)[None, :]


def accident_prob(g):
    # alpha(j, d) para d = 0..A-1 (Tabla 4).  shape (J, A)
    d = jnp.arange(g.a_max)[None, :]
    return jax.nn.sigmoid(jnp.asarray(g.acc_int)[:, None] + jnp.asarray(g.acc_age)[:, None] * d)


def sell_cost(g):
    # Ts(a), a = 1..A-1: tc_sell_inspect en edades pares >= inspect_age_min
    # jnp.where (no np.where): tc_sell puede ser un parámetro dinámico (estimación)
    a = np.arange(1, g.a_max)
    inspect = jnp.asarray((a % 2 == 0) & (a >= g.inspect_age_min))
    return jnp.where(inspect, g.tc_sell_inspect, g.tc_sell)


def emax(values, sigma):
    return sigma * logsumexp(jnp.stack(values, axis=0) / sigma, axis=0)


def choice_probs(values, sigma):
    v = jnp.stack(values, axis=0) / sigma
    return jnp.exp(v - logsumexp(v, axis=0, keepdims=True))


# Valores ______________________________________________________________

def continuation_values(EV, g):
    # beta * E[EV(x') | h].  cv_used (J, A-1) para d = 1..A-1; cv_new (J,)
    J, A, n_act, n = dims(g)
    act, term, none = split_states(EV, g)
    al = accident_prob(g)
    surv = (1.0 - al[:, 1:A - 1]) * act[:, 1:] + al[:, 1:A - 1] * term[:, None]   # d = 1..A-2
    cv_used = g.beta * jnp.concatenate([surv, term[:, None]], axis=1)          # d = A-1 -> term
    cv_new = g.beta * ((1.0 - al[:, 0]) * act[:, 0] + al[:, 0] * term)
    return cv_used, cv_new


class Values(NamedTuple):
    keep: jnp.ndarray
    purge_act: jnp.ndarray
    trade_act: jnp.ndarray
    purge_term: jnp.ndarray
    trade_term: jnp.ndarray
    stay_none: jnp.ndarray
    trade_none: jnp.ndarray
    buy: jnp.ndarray          # (J*(A-1) + J,)
    scrap: jnp.ndarray        # (J, A-1) valor de chatarrear
    sell: jnp.ndarray         # (J, A-1) valor de vender


def choice_values(EV, P, g):
    u = flow_utility(g)
    cv_used, cv_new = continuation_values(EV, g)
    act, term, none = split_states(EV, g)
    p_new, p_scrap = jnp.asarray(g.p_new), jnp.asarray(g.p_scrap)

    keep = u[:, 1:] + cv_used
    scrap = jnp.broadcast_to(g.mu * p_scrap[:, None], P.shape)
    sell = g.mu * P - sell_cost(g)[None, :]
    disposal = emax([scrap, sell], g.sigma_sell)

    buy_used = u[:, 1:] - g.mu * P - g.tc_buy + cv_used
    buy_new = u[:, 0] - g.mu * p_new - g.tc_buy + cv_new
    buy = jnp.concatenate([jnp.ravel(buy_used), buy_new])
    iv_buy = g.sigma_trade * logsumexp(buy / g.sigma_trade)

    no_car = g.u_none + g.beta * none
    term_tc = g.tc_buy_nocar if g.term_pays_nocar else 0.0
    return Values(
        keep=keep,
        purge_act=disposal + no_car,
        trade_act=disposal + iv_buy,
        purge_term=g.mu * p_scrap + no_car,
        trade_term=g.mu * p_scrap - term_tc + iv_buy,
        stay_none=no_car,
        trade_none=-g.tc_buy_nocar + iv_buy,
        buy=buy, scrap=scrap, sell=sell,
    )


def T_raw(EV, P, g):
    # Sin jit: acepta g estático (GParams) o dinámico (theta.GModel)
    v = choice_values(EV, P, g)
    return stack_states(emax([v.keep, v.purge_act, v.trade_act], g.sigma),
                        emax([v.purge_term, v.trade_term], g.sigma),
                        emax([v.stay_none, v.trade_none], g.sigma))


T = jax.jit(T_raw, static_argnames="g")


def solve_bellman(P, g, EV_init=None):
    # SA hasta sa_tol y luego Newton-Kantorovich con dT/dEV = beta M (Lema L1)
    from transitions import transition_matrix
    J, A, n_act, n = dims(g)
    EV = jnp.zeros(n) if EV_init is None else EV_init
    fpi = FixedPointIteration(fixed_point_fun=lambda V: T(V, P, g), implicit_diff=False,
                              maxiter=g.sa_max_iter, tol=g.sa_tol)
    EV = fpi.run(EV).params
    I = jnp.eye(n)
    for _ in range(g.nk_max_iter):
        TEV = T(EV, P, g)
        if float(jnp.max(jnp.abs(TEV - EV))) < g.vfi_tol:
            break
        EV = EV - jnp.linalg.solve(I - g.beta * transition_matrix(EV, P, g), EV - TEV)
    return EV
