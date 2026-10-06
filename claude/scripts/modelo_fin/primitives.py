# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/primitives.py
# Goal:           Utilidad, costos, transición de s y shocks
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
Los shocks entran SOLO por `emax` y `choice_probs` y por los sigmas de Params.
Para otra distribución, cambia esas dos funciones (y revisa el jacobiano de NK).

La transición de s se escribe en dos niveles:
- `s_transition_rows(mean, sd, grid)`: discretización genérica, sin `g`.  La usa
  también la verosimilitud, donde los parámetros de s son dinámicos.
- `s_transition(g)`: el tensor completo F[r, j, d, s, s'] del modelo.
'''

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

from utils import make_s_grid


# Utilidad y costos ________________________________________________________________

def flow_utility(g):
    # u(j, d, s) para d = 0..A-1 (d = 0 coche nuevo).  shape (J, A, S)
    d = jnp.arange(g.a_max)[None, :, None]
    s = make_s_grid(g)[None, None, :]
    u0 = jnp.asarray(g.u0)[:, None, None]
    u1 = jnp.asarray(g.u1)[:, None, None]
    u2 = jnp.asarray(g.u2)[:, None, None]
    return u0 + u1 * d + u2 * d ** 2 + g.u_s * s


def sell_cost(g):
    # Ts(a) en utils para el coche que se vende, a = 1..A-1.  shape (A-1,)
    # Gillingham Tabla 5: más caro en años de inspección (pares >= inspect_age_min)
    # jnp.where (no np.where): tc_sell puede ser dinámico (theta.Model)
    a = np.arange(1, g.a_max)
    inspect = jnp.asarray((a % 2 == 0) & (a >= g.inspect_age_min))
    return jnp.where(inspect, g.tc_sell_inspect, g.tc_sell)


# Transición de s ________________________________________________________________

def s_transition_rows(mean, sd, grid):
    # Pr(s' = grid_k | media, sd) para cada media; shape mean.shape + (S,).
    # Discretización por intervalos alrededor de cada punto del grid; las colas
    # se acumulan en los extremos (ojo: ahí E[eta|s] = 0 deja de cumplirse).
    # Bordes finitos y lejanos en vez de +-inf: con +-inf la derivada respecto a sd da
    # 0 * inf = nan (s_sigma se estima).  A 1e3 de distancia la cdf es exactamente 0 o 1,
    # así que los valores no cambian.
    mid = 0.5 * (grid[1:] + grid[:-1])
    edges = jnp.concatenate([grid[:1] - 1e3, mid, grid[-1:] + 1e3])
    cdf = norm.cdf((edges - mean[..., None]) / sd)
    return jnp.diff(cdf, axis=-1)


def s_mean_next(g):
    # m(r, j, d, s): la "m(y, s)" de Hu & Xin.  shape (2, J, A, S)
    r = jnp.arange(2)[:, None, None, None]
    c = jnp.asarray(g.s_const)[None, :, None, None]
    d = jnp.arange(g.a_max)[None, None, :, None]
    s = make_s_grid(g)[None, None, None, :]
    return c + g.s_age * d + g.s_persist * s - g.s_repair * r


def s_transition(g):
    # F[r, j, d, s, s'] = Pr(s' | s, j, d, r, sobrevive).  shape (2, J, A, S, S)
    return s_transition_rows(s_mean_next(g), g.s_sigma, make_s_grid(g))


# Shocks ________________________________________________________________

def emax(values, sigma):
    # E max_k {v_k + sigma * eps_k}, eps EV1 iid (sin la constante de Euler,
    # como en Gillingham; solo desplaza EV en sigma*0.5772/(1-beta)).
    return sigma * logsumexp(jnp.stack(values, axis=0) / sigma, axis=0)


def choice_probs(values, sigma):
    # Probabilidades logit de cada alternativa (eje 0 = alternativa)
    v = jnp.stack(values, axis=0) / sigma
    return jnp.exp(v - logsumexp(v, axis=0, keepdims=True))
