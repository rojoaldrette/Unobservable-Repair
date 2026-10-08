# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/gillingham/utils.py
# Goal:           Layout de estados, log-sumas y utilidades por tipo de hogar
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           07/10/2026
#
# _____________________________________________________________________________
'''
Layout (truco de Gillingham, sec. 3.4: el hueco del terminal lo ocupa el coche nuevo):

    X (inicio de periodo)       = [ act(j, a), a = 1..A-1 | term(j) | none ]
    H (tenencia post-comercio)  = [ used(j, d), d = 1..A-1 | new(j)  | none ]
    n = J (A - 1) + J + 1       (51 con J = 2, A = 25)

Así la matriz de comercio Ω (X -> H) y la física Q (H -> X) son cuadradas y el índice
de act(j, a) en X es el de used(j, a) en H (quedarse el coche = diagonal).
'''

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from params import TYPE_KEYS

jax.config.update("jax_enable_x64", True)


def dims(cfg):
    J, A = cfg.J, cfg.a_max
    n_act = J * (A - 1)
    return J, A, n_act, n_act + J + 1


def split_states(v, cfg):
    J, A, n_act, n = dims(cfg)
    return v[:n_act].reshape(J, A - 1), v[n_act:n_act + J], v[-1]


def stack_states(act, term, none):
    return jnp.concatenate([jnp.ravel(act), jnp.ravel(term), jnp.reshape(none, (1,))])


def emax(values, sigma):
    # E max (log-suma) de alternativas con shocks EV1 de escala sigma
    return sigma * logsumexp(jnp.stack(values) / sigma, axis=0)


def choice_probs(values, sigma):
    v = jnp.stack(values) / sigma
    return jnp.exp(v - logsumexp(v, axis=0, keepdims=True))


# Tipos de hogar ______________________________________________________________

def type_axes(th):
    # in_axes para jax.vmap sobre tipos: eje 0 en los campos por tipo, None en los comunes
    return {k: (0 if k in TYPE_KEYS else None) for k in th}


def type_theta(th, t):
    return {k: (v[t] if k in TYPE_KEYS else v) for k, v in th.items()}


def initial_prices(th, cfg, dep=0.87):
    # Depreciación exponencial (factor de la Tabla 3) con piso en p_scrap: arranque del solver
    a = jnp.arange(1, cfg.a_max)[None, :]
    return jnp.maximum(th["p_new"][:, None] * dep ** a, th["p_scrap"][:, None])
