# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/equilibrium.py
# Goal:           Equilibrio rápido con parámetros dinámicos y derivadas implícitas
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           04/10/2026
#
# _____________________________________________________________________________
'''
Para estimar hay que resolver el equilibrio en cada evaluación de la verosimilitud
(el "doble" de DNFXP).  Aquí EV y P se resuelven juntos: z = (EV, P) y

    F(z, theta) = [ EV - T(EV, P)  ;  ED_log(EV, P) ] = 0,      n + J(A-1) ecuaciones

con Newton sobre el sistema conjunto y jacobiano denso por jacfwd (148 x 148 con
a_max = 25).  Todo está jiteado con m = GModel(g, th) como pytree: un valor nuevo de th
no recompila.

Derivada del equilibrio respecto a los parámetros (función implícita, como en la sec.
5.1 del paper):  dz/dx = -F_z^{-1} F_x.
'''

import numpy as np
import jax
import jax.numpy as jnp

from utils import dims, initial_prices
from bellman import T_raw
from ED import excess_demand_log


def split_z(z, g):
    J, A, n_act, n = dims(g)
    return z[:n], z[n:].reshape(J, A - 1)


def residual(z, m):
    EV, P = split_z(z, m)
    return jnp.concatenate([EV - T_raw(EV, P, m), jnp.ravel(excess_demand_log(EV, P, m))])


@jax.jit
def _sa(EV, P, m):
    # 1,000 iteraciones de aproximaciones sucesivas: error ~ beta^1000 de la inicial
    return jax.lax.fori_loop(0, 1_000, lambda i, V: T_raw(V, P, m), EV)


@jax.jit
def _newton(z, m):
    F = residual(z, m)
    Jz = jax.jacfwd(residual)(z, m)
    return F, -jnp.linalg.solve(Jz, F)


@jax.jit
def _resid(z, m):
    return residual(z, m)


def initial_z(m):
    J, A, n_act, n = dims(m)
    P = initial_prices(m)
    EV = _sa(jnp.zeros(n), P, m)
    return jnp.concatenate([EV, jnp.ravel(P)])


def solve(m, z0=None, tol=1e-10, max_iter=50, verbose=False):
    # Newton con búsqueda de línea sobre ||F||.  Devuelve (z, convergió, iteraciones).
    z = initial_z(m) if z0 is None else jnp.asarray(z0)
    for it in range(max_iter):
        F, dz = _newton(z, m)
        err = float(jnp.max(jnp.abs(F)))
        if verbose:
            print(f"    Newton {it}: max|F| = {err:.2e}")
        if err < tol:
            return z, True, it
        nrm = float(jnp.linalg.norm(F))
        lam, ok = 1.0, False
        for _ in range(30):
            z_new = z + lam * dz
            F_new = _resid(z_new, m)
            if bool(jnp.all(jnp.isfinite(F_new))) and float(jnp.linalg.norm(F_new)) < nrm:
                ok = True
                break
            lam *= 0.5
        if not ok:
            # Newton conjunto atorado: resolver EV dados los P y seguir
            EV, P = split_z(z, m)
            z_new = jnp.concatenate([_sa(EV, P, m), jnp.ravel(P)])
        z = z_new
    return z, False, max_iter


def solve_or_restart(m, z0=None, **kw):
    # Arranque en caliente; si no converge, desde cero
    z, ok, it = solve(m, z0, **kw)
    if not ok and z0 is not None:
        z, ok, it = solve(m, None, **kw)
    return z, ok, it
