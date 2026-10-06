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

'''
Dos versiones de lo mismo:

- **Sin matrices (la que se usa):** `holding_apply`, `holding_rapply`, `M_apply`,
  `M_rapply` y `stationary_q` trabajan con el kernel de s (J, A-1, S, S) y con la
  estructura de M = ΩQ:
      Ω_keep diagonal,  Ω_trade = trade ⊗ buy (rango 1),  Ω_purge = una columna (none).
  Memoria O(J A S²) en lugar de O(n²): con a_max = 25 y n_s = 100, 12 MB contra 415 MB
  por matriz.
- **Densas (solo pruebas a tamaño chico):** `physical_matrices`, `holding_transition`,
  `trade_matrices`, `transition_matrix`, `stationary_distribution`.

Q (de tenencia h a estado x') sin matriz:
    usado (j, d, s), d <= A-2:  act (j, d+1, s')  con prob (1 - s) F_mix(s' | j, d, s)
                                term (j)          con prob s
    usado (j, A-1, s):          term (j)          con prob 1
    nuevo j:                    act (j, 1, s')    con prob (1 - s0) F_0(s' | j, 0, s0)
                                term (j)          con prob s0
    none:                       none
F_mix = (1 - p_rep(h)) F_0 + p_rep(h) F_1  (la etapa 2 integrada).
'''

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from utils import dims, make_s_grid, s_new_index, split_states, stack_states
from primitives import s_transition
from probabilities import ccps


# Sin matrices ______________________________________________________________

class Kernel(NamedTuple):
    used: jnp.ndarray     # (J, A-1, S, S): usado (j, d, s) -> act (j, d+1, s'), ya con (1 - s); 0 en d = A-1
    p_term: jnp.ndarray   # (A-1, S): Pr(usado -> term)
    new: jnp.ndarray      # (J, S): nuevo j -> act (j, 1, s'), ya con (1 - s0)
    s0: jnp.ndarray       # (J,): Pr(nuevo j -> term)


def holding_kernel(c, g):
    J, A, S, n_act, n = dims(g)
    F = s_transition(g)
    grid = make_s_grid(g)
    i0 = s_new_index(g)
    pr, _, _ = split_states(c.repair, g)
    F_mix = (1.0 - pr)[..., None] * F[0, :, 1:] + pr[..., None] * F[1, :, 1:]
    alive = (jnp.arange(1, A) < A - 1)[None, :, None, None]
    used = jnp.where(alive, (1.0 - grid)[None, None, :, None] * F_mix, 0.0)
    is_last = (jnp.arange(A - 1) == A - 2)[:, None]
    p_term = jnp.where(is_last, 1.0, grid[None, :])
    s0 = grid[i0]                                              # (J,)
    F_new = F[0, jnp.arange(J), 0, i0, :]                      # (J, S)
    return Kernel(used=used, p_term=p_term, new=(1.0 - s0)[:, None] * F_new, s0=s0)


def holding_apply(v, K, g):
    # (Q v)(h) = E[v(x') | h]:  v sobre X  ->  vector sobre H.  Es el producto Q @ v.
    J, A, S, n_act, n = dims(g)
    act, term, none = split_states(v, g)
    act_next = jnp.concatenate([act[:, 1:], jnp.zeros((J, 1, S))], axis=1)
    used = jnp.einsum("jdst,jdt->jds", K.used, act_next) + K.p_term[None] * term[:, None, None]
    new = jnp.einsum("jt,jt->j", K.new, act[:, 0, :]) + K.s0 * term
    return stack_states(used, new, none)


def holding_rapply(w, K, g):
    # (w Q)(x') = sum_h w(h) Q(h, x'):  w sobre H  ->  vector sobre X.  Es w @ Q.
    J, A, S, n_act, n = dims(g)
    w_used, w_new, w_none = split_states(w, g)
    age1 = w_new[:, None] * K.new
    older = jnp.einsum("jds,jdst->jdt", w_used[:, :A - 2], K.used[:, :A - 2])
    act = jnp.concatenate([age1[:, None, :], older], axis=1)
    term = jnp.sum(w_used * K.p_term[None], axis=(1, 2)) + w_new * K.s0
    return stack_states(act, term, w_none)


def M_apply(v, c, K, g):
    # M v con M = Ω Q, sin armar M.  (Ω Q v)(x) = keep(x) (Qv)(x) + trade(x) <buy, Qv> + purge(x) v(none)
    Qv = holding_apply(v, K, g)
    return c.keep * Qv + c.trade * (c.buy @ Qv) + c.purge * v[-1]


def holding_distribution(q, c):
    # q_hold = q Ω = keep ⊙ q + <q, trade> buy + <q, purge> e_none  (sobre H)
    e_none = jnp.concatenate([jnp.zeros(q.shape[0] - 1), jnp.ones(1)])
    return c.keep * q + (q @ c.trade) * c.buy + (q @ c.purge) * e_none


def M_rapply(q, c, K, g):
    # q M sin armar M:  (q Ω) Q
    return holding_rapply(holding_distribution(q, c), K, g)


def stationary_q(c, g):
    # q = q M, sum(q) = 1, exacta y sin resolver sistemas.
    # Todo coche activo viene de una compra, y las compras se reparten según `buy`, que
    # no depende de x.  Con una masa compradora m = 1:
    #   act(j, 1, .)   = buy_new(j) K.new(j, .)
    #   hold(j, d, .)  = act(j, d, .) keep(j, d, .) + buy_used(j, d, .)
    #   act(j, d+1, .) = hold(j, d, .) @ K.used(j, d)            (recursión en la edad)
    #   term(j)        = sum_{d,s} hold p_term + buy_new(j) s0
    #   none           = (lo que entra por purge) / Pr(trade | none)
    # Todo es lineal en m; m sale de normalizar.  Sólo suma productos no negativos, así
    # que q tiene precisión relativa de máquina aun en celdas de 1e-15 (un solve denso o
    # GMRES solo dan precisión absoluta).
    J, A, S, n_act, n = dims(g)
    K = holding_kernel(c, g)
    keep, _, _ = split_states(c.keep, g)
    purge_act, purge_term, _ = split_states(c.purge, g)
    _, _, trade_none = split_states(c.trade, g)
    b_used = c.buy[:n_act].reshape(J, A - 1, S)
    b_new = c.buy[n_act:n_act + J]

    def step(act_d, xs):
        keep_d, b_d, K_d = xs
        nxt = jnp.einsum("js,jst->jt", act_d * keep_d + b_d, K_d)
        return nxt, act_d

    xs = (jnp.moveaxis(keep[:, :A - 2], 1, 0), jnp.moveaxis(b_used[:, :A - 2], 1, 0),
          jnp.moveaxis(K.used[:, :A - 2], 1, 0))
    last, acts = jax.lax.scan(step, b_new[:, None] * K.new, xs)
    act = jnp.concatenate([jnp.moveaxis(acts, 0, 1), last[:, None, :]], axis=1)   # (J, A-1, S)

    hold = act * keep + b_used
    term = jnp.sum(hold * K.p_term[None], axis=(1, 2)) + b_new * K.s0
    none = (jnp.sum(act * purge_act) + jnp.sum(term * purge_term)) / trade_none
    q = stack_states(act, term, none)
    return q / jnp.sum(q)


# Densas (solo pruebas) ______________________________________________________________

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
    F_new = (1.0 - s0)[:, None] * F[0, jnp.arange(J), 0, i0, :]
    new_to_act = jnp.einsum("jk,e,jt->jket", eyeJ, first, F_new).reshape(J, n_act)
    new_to_term = jnp.diag(s0)

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
