# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/ED.py
# Goal:           Exceso de demanda y equilibrio estacionario (Gillingham sec. 3.4–3.5)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
    D(j, a) = trade_mass * Pr(comprar (j, a) | trade)
    S(j, a) = q(j, a) * (1 - Pr(keep)) * (1 - Pr(chatarrear))     (lo chatarreado sale)
    ED_log  = log D - log S = 0,   J(A-1) incógnitas (72 con a_max = 25)

El sistema es chico: Newton con jacobiano denso.  El efecto de P sobre EV entra por
la función implícita, dEV/dP = (I - beta M)^{-1} dT/dP.
'''

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from utils import dims, split_states, initial_prices
from bellman import T, choice_values, solve_bellman, accident_prob
from probabilities import ccps_raw
from transitions import transition_from_ccps, stationary_distribution


class GMarket(NamedTuple):
    q: jnp.ndarray
    keep: jnp.ndarray        # (J, A-1)
    trade: jnp.ndarray
    purge: jnp.ndarray
    scrap: jnp.ndarray       # (J, A-1) Pr(chatarrear | se deshace)
    buy_used: jnp.ndarray    # (J, A-1)
    buy_new: jnp.ndarray     # (J,)
    trade_mass: jnp.ndarray
    demand: jnp.ndarray
    supply: jnp.ndarray
    endo_scrap: jnp.ndarray  # (J, A-1) Pr(chatarreo endógeno | x) = (1 - keep) * scrap
    accident: jnp.ndarray    # (J, A)   alpha(j, d), d = 0..A-1


def market_components(EV, P, g):
    J, A, n_act, n = dims(g)
    c = ccps_raw(EV, P, g)
    q = stationary_distribution(transition_from_ccps(c, g))
    qa, _, _ = split_states(q, g)
    pk, _, _ = split_states(c.keep, g)
    pt, _, _ = split_states(c.trade, g)
    pp, _, _ = split_states(c.purge, g)
    buy_used = c.buy[:n_act].reshape(J, A - 1)
    tm = q @ c.trade
    return GMarket(q=q, keep=pk, trade=pt, purge=pp, scrap=c.scrap, buy_used=buy_used,
                   buy_new=c.buy[n_act:n_act + J], trade_mass=tm, demand=tm * buy_used,
                   supply=qa * (1.0 - pk) * (1.0 - c.scrap),
                   endo_scrap=(1.0 - pk) * c.scrap, accident=accident_prob(g))


def excess_demand_log(EV, P, g):
    J, A, n_act, n = dims(g)
    c = ccps_raw(EV, P, g)
    q = stationary_distribution(transition_from_ccps(c, g))
    qa, _, _ = split_states(q, g)
    pk, _, _ = split_states(c.keep, g)
    v = choice_values(EV, P, g)
    log_buy = (v.buy / g.sigma_trade - logsumexp(v.buy / g.sigma_trade))[:n_act].reshape(J, A - 1)
    log_D = jnp.log(q @ c.trade) + log_buy
    log_S = jnp.log(jnp.maximum(qa * (1.0 - pk) * (1.0 - c.scrap), 1e-300))
    return log_D - log_S


class GEquilibrium(NamedTuple):
    P: jnp.ndarray
    EV: jnp.ndarray
    ed: jnp.ndarray
    converged: bool
    iters: int


def solve_equilibrium(g, P_init=None, verbose=False):
    P = initial_prices(g) if P_init is None else jnp.asarray(P_init)
    shp, m = P.shape, P.size
    EV = solve_bellman(P, g)
    ed = excess_demand_log(EV, P, g)
    converged, it = False, 0
    for it in range(g.ed_max_iter):
        err = float(jnp.max(jnp.abs(ed)))
        if verbose:
            print(f"  Newton {it}: max|ED_log| = {err:.2e}")
        if err < g.ed_tol:
            converged = True
            break
        from transitions import transition_matrix
        M = transition_matrix(EV, P, g)
        J_P = jax.jacfwd(lambda P_: excess_demand_log(EV, P_, g))(P).reshape(m, m)
        J_EV = jax.jacfwd(lambda E_: excess_demand_log(E_, P, g))(EV).reshape(m, -1)
        T_P = jax.jacfwd(lambda P_: T(EV, P_, g))(P).reshape(-1, m)
        dEV = jnp.linalg.solve(jnp.eye(EV.shape[0]) - g.beta * M, T_P)
        dP = -jnp.linalg.solve(J_P + J_EV @ dEV, ed.ravel()).reshape(shp)
        lam = 1.0
        for _ in range(20):
            EV_new = solve_bellman(P + lam * dP, g, EV)
            ed_new = excess_demand_log(EV_new, P + lam * dP, g)
            if bool(jnp.all(jnp.isfinite(ed_new))) and float(jnp.max(jnp.abs(ed_new))) < err:
                break
            lam *= 0.5
        P, EV, ed = P + lam * dP, EV_new, ed_new
    if verbose:
        print(f"  equilibrio: convergió={converged}, max|ED_log|={float(jnp.max(jnp.abs(ed))):.2e}")
    return GEquilibrium(P=P, EV=EV, ed=ed, converged=converged, iters=it)
