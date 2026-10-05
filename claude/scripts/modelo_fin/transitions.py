# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/transitions.py
# Goal:           Matrices de comercio (Ω), físicas (Q), M = ΩQ y distribución estacionaria
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________

from functools import partial

import jax
import jax.numpy as jnp

from utils import dims, make_s_grid, s_new_index, split_states
from primitives import s_transition
from probabilities import ccps


@partial(jax.jit, static_argnames="g")
def physical_matrices(g):
    # Q_r (n x n), de tenencia h (layout H) a estado x' (layout X).
    # Q_0: sin reparar; Q_1: reparado (solo difiere en filas de usados).
    J, A, S, n_act, n = dims(g)
    F = s_transition(g)
    grid = make_s_grid(g)
    i0 = s_new_index(g)
    s0 = grid[i0]
    eyeJ = jnp.eye(J)

    # usado (j,d,s) -> act (j,d+1,s'):  (1-s) F_r[j,d,s,s']   (shift mata d = A-1)
    shift = jnp.eye(A - 1, k=1)
    surv = (1.0 - grid)[None, None, None, :, None] * F[:, :, 1:]
    act_to_act = jnp.einsum("jk,de,rjdst->rjdsket", eyeJ, shift, surv).reshape(2, n_act, n_act)

    # usado -> term: prob s, o 1 si d = A-1
    is_last = (jnp.arange(A - 1) == A - 2)[:, None]
    p_term = jnp.where(is_last, 1.0, grid[None, :])                 # (A-1, S)
    act_to_term = (p_term[None, :, :, None] * eyeJ[:, None, None, :]).reshape(n_act, J)

    # nuevo j -> act (j, 1, s'), o term con prob s0
    first = (jnp.arange(A - 1) == 0).astype(grid.dtype)
    new_to_act = jnp.einsum("jk,e,jt->jket", eyeJ, first,
                            (1.0 - s0) * F[0, :, 0, i0, :]).reshape(J, n_act)
    new_to_term = s0 * eyeJ

    def assemble(r):
        top = jnp.concatenate([act_to_act[r], act_to_term, jnp.zeros((n_act, 1))], axis=1)
        mid = jnp.concatenate([new_to_act, new_to_term, jnp.zeros((J, 1))], axis=1)
        bot = jnp.concatenate([jnp.zeros((1, n_act + J)), jnp.ones((1, 1))], axis=1)
        return jnp.concatenate([top, mid, bot], axis=0)

    return assemble(0), assemble(1)


def holding_transition(c, g):
    # Q(h -> x') ya integrando la etapa 2:  (1 - p_rep(h)) Q_0 + p_rep(h) Q_1.  (n x n)
    Q0, Q1 = physical_matrices(g)
    return (1.0 - c.repair)[:, None] * Q0 + c.repair[:, None] * Q1


def trade_matrices(c, g):
    # Ω desagregada (n x n, de X a H).  Ω_total = suma de las tres; filas suman 1.
    # La reparación no está aquí: es de la etapa 2 y vive en holding_transition.
    J, A, S, n_act, n = dims(g)
    e_none = jnp.concatenate([jnp.zeros(n - 1), jnp.ones(1)])
    return dict(
        keep=jnp.diag(c.keep),                 # te quedas con el mismo coche
        trade=jnp.outer(c.trade, c.buy),       # vendes (o entregas el terminal) y compras h
        purge=jnp.outer(c.purge, e_none),      # te quedas sin coche
    )


def transition_from_ccps(c, g):
    # M = Ω Q,  Q = diag(1 - p_rep) Q_0 + diag(p_rep) Q_1   (n x n, X -> X)
    # Se arma sin productos densos n^3: Ω_keep es diagonal, Ω_trade es de rango 1
    # y Ω_purge solo llena la columna none.
    Q = holding_transition(c, g)
    J, A, S, n_act, n = dims(g)
    e_none = jnp.concatenate([jnp.zeros(n - 1), jnp.ones(1)])
    return (c.keep[:, None] * Q
            + c.trade[:, None] * (c.buy @ Q)[None, :]
            + c.purge[:, None] * e_none[None, :])


@partial(jax.jit, static_argnames="g")
def transition_matrix(EV, P, g):
    return transition_from_ccps(ccps(EV, P, g), g)


def stationary_distribution(M):
    # q = q M,  sum(q) = 1   (Gillingham ec. 29, dado P)
    n = M.shape[0]
    A_ = jnp.concatenate([(M.T - jnp.eye(n))[:-1], jnp.ones((1, n))], axis=0)
    b = jnp.concatenate([jnp.zeros(n - 1), jnp.ones(1)])
    return jnp.linalg.solve(A_, b)


def observed_s_transition(c, g):
    # Lo que ve el econometrista para un usado h que sobrevive (Hu & Xin ec. 6.4):
    #   f(s'|j,d,s) = (1 - p_rep) F_0 + p_rep F_1
    # Con la decisión secuencial el peso es el mismo si h se conservó o se compró,
    # así que keepers y compradores de usados entran a la misma mezcla.
    # shape (J, A-1, S, S).
    F = s_transition(g)[:, :, 1:]
    pr, _, _ = split_states(c.repair, g)
    return (1.0 - pr)[..., None] * F[0] + pr[..., None] * F[1]
