# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/utils.py
# Goal:           Dimensiones, grids y layout de estados
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
Layout de estados (inicio de periodo, antes de comerciar), vector de largo n:
    X = [ act(j, a, s)  a=1..A-1  |  term(j)  |  none ]
Layout de tenencias post-comercio (lo que manejas este periodo), mismo largo:
    H = [ used(j, d, s) d=1..A-1  |  new(j)   |  none ]
(el truco de Gillingham: el hueco del terminal se recicla para el coche nuevo)
n = J*(A-1)*S + J + 1   (3*6*100 + 3 + 1 = 1804 con los defaults)

Un índice i < n_act es el mismo coche (j, a, s) en X y en H: quedarse el coche
manda x = i a h = i.
'''

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)   # tolerancias de 1e-12 necesitan float64


def dims(g):
    J, A, S = g.n_brands, g.a_max, g.n_s
    n_act = J * (A - 1) * S
    return J, A, S, n_act, n_act + J + 1


def make_s_grid(g):
    return jnp.linspace(g.s_min, g.s_max, g.n_s)


def s_new_index(g):
    # Python int (estático): índice del grid más cercano a s_new
    grid = np.linspace(g.s_min, g.s_max, g.n_s)
    return int(np.argmin(np.abs(grid - g.s_new)))


def split_states(x, g):
    # vector de largo n  ->  (act (J,A-1,S), term (J,), none ())
    J, A, S, n_act, n = dims(g)
    return x[:n_act].reshape(J, A - 1, S), x[n_act:n_act + J], x[-1]


def stack_states(act, term, none):
    return jnp.concatenate([jnp.ravel(act), jnp.ravel(term), jnp.reshape(none, (1,))])


def decode_states(idx, g):
    # índices (numpy) -> (tipo, j, a, s_idx).  tipo: 0 = act/used, 1 = term/new, 2 = none.
    # Sirve igual para X y para H (en H, tipo 1 es coche nuevo y a = 0).
    J, A, S, n_act, n = dims(g)
    idx = np.asarray(idx)
    kind = np.where(idx < n_act, 0, np.where(idx < n_act + J, 1, 2))
    act = np.minimum(idx, n_act - 1)
    j = np.where(kind == 0, act // ((A - 1) * S), np.where(kind == 1, idx - n_act, -1))
    a = np.where(kind == 0, (act // S) % (A - 1) + 1, np.where(kind == 1, 0, -1))
    s = np.where(kind == 0, act % S, -1)
    return kind, j, a, s


def initial_prices(g):
    # P0(j, a, s) = p_new_j * dep^a * (1 - s), acotado abajo por la chatarra
    a = jnp.arange(1, g.a_max)[None, :, None]
    s = make_s_grid(g)[None, None, :]
    p_new = jnp.asarray(g.p_new)[:, None, None]
    p_scrap = jnp.asarray(g.p_scrap)[:, None, None]
    return jnp.maximum(p_new * g.dep_factor ** a * (1.0 - s), p_scrap)
