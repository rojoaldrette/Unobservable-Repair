# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/transitions.py
# Goal:           Q (H -> X), M = ΩQ y distribución estacionaria (ec. 29)
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

from utils import dims
from bellman import accident_prob
from probabilities import ccps


@partial(jax.jit, static_argnames="g")
def physical_matrix(g):
    # usado (j, d) -> act (j, d+1) con 1 - alpha, term con alpha (d = A-1: term seguro)
    # nuevo j -> act (j, 1) con 1 - alpha(j, 0), term con alpha(j, 0);  none -> none
    J, A, n_act, n = dims(g)
    al = accident_prob(g)
    eyeJ = jnp.eye(J)
    surv = jnp.concatenate([1.0 - al[:, 1:A - 1], jnp.zeros((J, 1))], axis=1)       # (J, A-1)
    shift = jnp.eye(A - 1, k=1)
    act_to_act = jnp.einsum("jk,de,jd->jdke", eyeJ, shift, surv).reshape(n_act, n_act)
    act_to_term = ((1.0 - surv)[:, :, None] * eyeJ[:, None, :]).reshape(n_act, J)
    first = (jnp.arange(A - 1) == 0).astype(al.dtype)
    new_to_act = jnp.einsum("jk,e,j->jke", eyeJ, first, 1.0 - al[:, 0]).reshape(J, n_act)
    new_to_term = al[:, 0][:, None] * eyeJ
    top = jnp.concatenate([act_to_act, act_to_term, jnp.zeros((n_act, 1))], axis=1)
    mid = jnp.concatenate([new_to_act, new_to_term, jnp.zeros((J, 1))], axis=1)
    bot = jnp.concatenate([jnp.zeros((1, n_act + J)), jnp.ones((1, 1))], axis=1)
    return jnp.concatenate([top, mid, bot], axis=0)


def transition_from_ccps(c, g):
    Q = physical_matrix(g)
    J, A, n_act, n = dims(g)
    e_none = jnp.concatenate([jnp.zeros(n - 1), jnp.ones(1)])
    return (c.keep[:, None] * Q
            + c.trade[:, None] * (c.buy @ Q)[None, :]
            + c.purge[:, None] * e_none[None, :])


@partial(jax.jit, static_argnames="g")
def transition_matrix(EV, P, g):
    return transition_from_ccps(ccps(EV, P, g), g)


def stationary_distribution(M):
    n = M.shape[0]
    A_ = jnp.concatenate([(M.T - jnp.eye(n))[:-1], jnp.ones((1, n))], axis=0)
    b = jnp.concatenate([jnp.zeros(n - 1), jnp.ones(1)])
    return jnp.linalg.solve(A_, b)
