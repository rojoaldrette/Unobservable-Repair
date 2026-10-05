# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/utils.py
# Goal:           Dimensiones y layout de estados de la réplica de Gillingham
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
Mismo truco de layout que modelo_fin, sin la dimensión s:
    X = [ act(j, a)  a=1..A-1 | term(j) | none ]     (inicio de periodo)
    H = [ used(j, d) d=1..A-1 | new(j)  | none ]     (tenencia después de comerciar)
    n = J*(A-1) + J + 1   (3*24 + 3 + 1 = 76 con a_max = 25)
'''

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def dims(g):
    J, A = g.n_brands, g.a_max
    n_act = J * (A - 1)
    return J, A, n_act, n_act + J + 1


def split_states(x, g):
    J, A, n_act, n = dims(g)
    return x[:n_act].reshape(J, A - 1), x[n_act:n_act + J], x[-1]


def stack_states(act, term, none):
    return jnp.concatenate([jnp.ravel(act), jnp.ravel(term), jnp.reshape(none, (1,))])


def initial_prices(g):
    a = jnp.arange(1, g.a_max)[None, :]
    p_new = jnp.asarray(g.p_new)[:, None]
    p_scrap = jnp.asarray(g.p_scrap)[:, None]
    return jnp.maximum(p_new * g.dep_factor ** a, p_scrap)
