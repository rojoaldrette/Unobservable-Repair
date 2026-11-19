# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/modelo_tesis/utils.py
# Goal:           Layout de estados, log-sumas, tipos de hogar y jacobiano por bloques
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
Layout (como Gillingham, con w):

    X (inicio de periodo)       = [ act(j, a, w), a = 1..A-1 | term(j) | none ]
    H (tenencia post-comercio)  = [ used(j, d, w), d = 1..A-1 | new(j)  | none ]
    n = J (A-1) W + J + 1        (3,411 con J = 2, A = 25, W = 71)

act(j, a, w) en X tiene el mismo índice que used(j, a, w) en H (quedarse el coche).
Un coche nuevo siempre tiene w = 0 (índice cfg.w0).
'''

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from params import TYPE_KEYS

jax.config.update("jax_enable_x64", True)


def dims(cfg):
    J, A, W = cfg.J, cfg.a_max, cfg.W
    n_act = J * (A - 1) * W
    return J, A, W, n_act, n_act + J + 1


def split_states(v, cfg):
    J, A, W, n_act, n = dims(cfg)
    return v[:n_act].reshape(J, A - 1, W), v[n_act:n_act + J], v[-1]


def stack_states(act, term, none):
    return jnp.concatenate([jnp.ravel(act), jnp.ravel(term), jnp.reshape(none, (1,))])


def emax(values, sigma):
    return sigma * logsumexp(jnp.stack(values) / sigma, axis=0)


def choice_probs(values, sigma):
    v = jnp.stack(values) / sigma
    return jnp.exp(v - logsumexp(v, axis=0, keepdims=True))


def type_axes(th):
    return {k: (0 if k in TYPE_KEYS else None) for k in th}


def type_theta(th, t):
    return {k: (v[t] if k in TYPE_KEYS else v) for k, v in th.items()}


def jacobian(fun, z, chunk):
    # Jacobiano denso de fun en z por bloques de `chunk` columnas (jvp sobre la
    # linealización): la memoria es O(chunk · tamaño de fun), no O(n · tamaño de fun).
    f0, jvp = jax.linearize(fun, z)
    n = z.shape[0]
    nb = -(-n // chunk)

    def block(b):
        V = jax.nn.one_hot(b * chunk + jnp.arange(chunk), n, dtype=z.dtype)   # filas fuera de rango = 0
        return jax.vmap(jvp)(V)

    Jt = jax.lax.map(block, jnp.arange(nb)).reshape(nb * chunk, -1)[:n]       # (n, m) = J^T
    return f0, Jt.T
