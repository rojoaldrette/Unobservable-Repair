# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/bellman.py
# Goal:           Valores por elección, operador de Bellman y solver
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
Quick comments:

1. 'g' es Params(), se controla desde main.  Es frozen + hashable, así que entra
   a jax.jit como argumento estático.  Los precios del mercado secundario P sí
   son argumento dinámico (el loop de equilibrio los va a mover).

2. Timing dentro del periodo (decisión secuencial, ver docs/estructura_decision.md):
   inicio en x -> etapa 1: {keep, purge, trade (+ qué h comprar)} -> tienes h ->
   etapa 2: shock nuevo, {reparar, no reparar} sobre h -> manejas h ->
   con prob s se descompone (-> term), si no s' = m(r,j,d,s) + eta, edad d+1.

3. Sin chatarreo endógeno.  Un coche activo solo sale de tus manos vendiéndolo
   (trade o purge, pagando Ts), por accidente o al llegar a la edad terminal.
'''

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxopt import FixedPointIteration

from utils import dims, make_s_grid, s_new_index, split_states, stack_states
from primitives import flow_utility, sell_cost, s_transition, emax


# Valores de continuación ______________________________________________________________

def continuation_values(EV, g):
    # CV_r(h) = beta * E[EV(x') | h, r].
    # Devuelve cv_used (2, J, A-1, S) y cv_new (J,)  (el nuevo nunca se repara)
    J, A, S, n_act, n = dims(g)
    act, term, none = split_states(EV, g)
    F = s_transition(g)
    grid = make_s_grid(g)

    # usados de edad 1..A-2: sobreviven a edad d+1 con s' ~ F
    EV_next = jnp.einsum("rjdst,jdt->rjds", F[:, :, 1:A - 1], act[:, 1:])
    survive = (1.0 - grid) * EV_next + grid * term[None, :, None, None]
    # edad A-1: llega a terminal con certeza
    last = jnp.broadcast_to(term[None, :, None, None], (2, J, 1, S))
    cv_used = g.beta * jnp.concatenate([survive, last], axis=2)

    i0 = s_new_index(g)
    s0 = grid[i0]
    cv_new = g.beta * ((1.0 - s0) * jnp.sum(F[0, :, 0, i0, :] * act[:, 0, :], axis=-1)
                       + s0 * term)
    return cv_used, cv_new


def repair_stage(EV, g):
    # Etapa 2, ya con el coche usado h en la mano: {no reparar, reparar}.
    # Devuelve (vals, W): vals = [v_r=0, v_r=1] (J, A-1, S) y W(h) = valor inclusivo,
    # que es lo que la etapa 1 ve de h.  No depende de P ni de cómo se llegó a h.
    cv_used, _ = continuation_values(EV, g)
    R = jnp.asarray(g.repair_price)[:, :, None]
    vals = [cv_used[0], -g.mu * R + cv_used[1]]
    return vals, emax(vals, g.sigma_repair)


# Valores por elección ______________________________________________________________

def buy_inclusive(buy_used, buy_new, g):
    # Nido de compra:  trade -> {(j, d) usados, nuevos j}  (escala sigma_trade)
    #                  (j, d) -> s                          (escala sigma_s)
    # Devuelve (iv_buy, I_jd): valor inclusivo de comprar y de cada nido (J, A-1).
    I_jd = g.sigma_s * logsumexp(buy_used / g.sigma_s, axis=-1)
    top = jnp.concatenate([jnp.ravel(I_jd), buy_new])
    return g.sigma_trade * logsumexp(top / g.sigma_trade), I_jd


def log_buy_probs(buy, g):
    # log Pr(comprar h | trade) sobre H sin la columna none, a partir del vector `buy`
    # de choice_values.  Usado: log p(j,d) + log p(s | j,d).  Nuevo: log p(j).
    J, A, S, n_act, n = dims(g)
    buy_used = buy[:n_act].reshape(J, A - 1, S)
    buy_new = buy[n_act:]
    iv, I_jd = buy_inclusive(buy_used, buy_new, g)
    lp_used = (I_jd - iv)[..., None] / g.sigma_trade + (buy_used - I_jd[..., None]) / g.sigma_s
    lp_new = (buy_new - iv) / g.sigma_trade
    return jnp.concatenate([jnp.ravel(lp_used), lp_new])


class Values(NamedTuple):
    keep: jnp.ndarray        # (J, A-1, S)   quedarse el coche (antes de la etapa 2)
    purge_act: jnp.ndarray   # (J, A-1, S)   vender y quedarse sin coche
    trade_act: jnp.ndarray   # (J, A-1, S)   vender y comprar (valor inclusivo del nido)
    purge_term: jnp.ndarray  # (J,)
    trade_term: jnp.ndarray  # (J,)
    stay_none: jnp.ndarray   # ()
    trade_none: jnp.ndarray  # ()
    buy: jnp.ndarray         # (J*(A-1)*S + J,)  valor de comprar h, sin términos del vendedor
    sell: jnp.ndarray        # (J, A-1, S)   lo que recibe quien vende su coche activo
    repair: tuple            # [v_r=0, v_r=1], cada uno (J, A-1, S), sobre H


def choice_values(EV, P, g):
    # P: precios del mercado secundario, shape (J, A-1, S), miles de DKK
    u = flow_utility(g)
    _, cv_new = continuation_values(EV, g)
    rep_vals, W = repair_stage(EV, g)
    act, term, none = split_states(EV, g)
    i0 = s_new_index(g)

    p_new = jnp.asarray(g.p_new)
    p_scrap = jnp.asarray(g.p_scrap)
    u_used = u[:, 1:, :]
    u_new = u[:, 0, i0]

    # Quedarse el coche: lo manejas y pasas a la etapa 2 con h = x
    keep = u_used + W

    # Deshacerse de un coche activo = venderlo (sin chatarreo endógeno)
    sell = g.mu * P - sell_cost(g)[None, :, None]

    # Comprar h: común a todos los estados (salvo una constante aditiva)
    buy_used = u_used - g.mu * P - g.tc_buy + W
    buy_new = u_new - g.mu * p_new - g.tc_buy + cv_new
    buy = jnp.concatenate([jnp.ravel(buy_used), buy_new])
    iv_buy = buy_inclusive(buy_used, buy_new, g)[0]

    no_car_next = g.u_none + g.beta * none

    return Values(
        keep=keep,
        purge_act=sell + no_car_next,
        trade_act=sell + iv_buy,
        purge_term=g.mu * p_scrap + no_car_next,
        trade_term=g.mu * p_scrap + iv_buy,
        stay_none=no_car_next,
        trade_none=-g.tc_buy_nocar + iv_buy,
        buy=buy,
        sell=sell,
        repair=rep_vals,
    )


# Operador de Bellman y solver ______________________________________________________________

# Operador de Bellman (Γ en Gillingham, ec. 9) sobre EV(x)
@partial(jax.jit, static_argnames="g")
def T(EV, P, g):
    v = choice_values(EV, P, g)
    ev_act = emax([v.keep, v.purge_act, v.trade_act], g.sigma)
    ev_term = emax([v.purge_term, v.trade_term], g.sigma)
    ev_none = emax([v.stay_none, v.trade_none], g.sigma)
    return stack_states(ev_act, ev_term, ev_none)


def solve_bellman(P, g, EV_init=None, verbose=False):
    # Successive approximations (jaxopt) hasta sa_tol, luego Newton-Kantorovich.
    # Jacobiano de T: dT/dEV = beta * M (Gillingham, Lema L1).  Sigue valiendo con la
    # etapa 2 porque d emax / d v son las CCPs en cada etapa, y la regla de la cadena
    # da Ω · [diag(1-p_rep) Q_0 + diag(p_rep) Q_1] = M (se verifica en tests.py).
    from transitions import transition_matrix   # import local: transitions -> probabilities -> bellman

    J, A, S, n_act, n = dims(g)
    EV = jnp.zeros(n) if EV_init is None else EV_init

    fpi = FixedPointIteration(fixed_point_fun=lambda V: T(V, P, g), implicit_diff=False,
                              maxiter=g.sa_max_iter, tol=g.sa_tol)
    sol = fpi.run(EV)
    EV = sol.params
    if verbose:
        print(f"SA: {int(sol.state.iter_num)} iteraciones, error {float(sol.state.error):.2e}")

    I = jnp.eye(n)
    for k in range(g.nk_max_iter):
        TEV = T(EV, P, g)
        err = float(jnp.max(jnp.abs(TEV - EV)))
        if verbose:
            print(f"NK {k}: sup|T(EV) - EV| = {err:.2e}")
        if err < g.vfi_tol:
            break
        M = transition_matrix(EV, P, g)
        EV = EV - jnp.linalg.solve(I - g.beta * M, EV - TEV)
    return EV
