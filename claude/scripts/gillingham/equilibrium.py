# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/equilibrium.py
# Goal:           Equilibrio rápido con T tipos de hogar, parámetros dinámicos
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           04/10/2026
#
# _____________________________________________________________________________
'''
Para estimar hay que resolver el equilibrio en cada evaluación de la verosimilitud
(el "doble" de DNFXP).  Con T tipos de hogar (fracciones f_t) y precios P comunes:

    z = (EV_0, ..., EV_{T-1}, P)
    F(z, theta) = [ EV_t - T_t(EV_t, P)  para cada t ;  ED_log(EV, P) ] = 0
    D(j, a) = sum_t f_t * trade_mass_t * Pr_t(comprar (j, a) | trade)
    S(j, a) = sum_t f_t * q_t(j, a) * (1 - keep_t) * (1 - scrap_t)
    ED_log  = log D - log S         (sec. 3.4-3.5 del paper)

Newton sobre el sistema conjunto con jacobiano denso por jacfwd (T n + J(A-1)
incógnitas: 148 con T = 1, 224 con T = 2 y a_max = 25).  Todo jiteado con
eco = Economy(g, th, f) como pytree: un th nuevo no recompila.  Con T = 1 es el mismo
sistema que ED.py.

Derivada del equilibrio respecto a los parámetros (función implícita, sec. 5.1):
dz/dx = -F_z^{-1} F_x.
'''

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from utils import dims, split_states, initial_prices
from bellman import T_raw, choice_values
from probabilities import ccps_raw
from transitions import transition_from_ccps, stationary_distribution


def split_z(z, eco):
    J, A, n_act, n = dims(eco.g)
    T = eco.n_types
    return z[:T * n].reshape(T, n), z[T * n:].reshape(J, A - 1)


def type_market(EV, P, m):
    # log D_t sin ponderar (log masa que compra + log Pr(comprar (j, a))) y S_t sin ponderar
    J, A, n_act, n = dims(m)
    c = ccps_raw(EV, P, m)
    q = stationary_distribution(transition_from_ccps(c, m))
    qa, _, _ = split_states(q, m)
    pk, _, _ = split_states(c.keep, m)
    v = choice_values(EV, P, m)
    log_buy = (v.buy / m.sigma_trade - logsumexp(v.buy / m.sigma_trade))[:n_act].reshape(J, A - 1)
    return jnp.log(q @ c.trade) + log_buy, qa * (1.0 - pk) * (1.0 - c.scrap)


def excess_demand_log(EVs, P, eco):
    log_D, S = [], 0.0
    for t in range(eco.n_types):
        ld, s = type_market(EVs[t], P, eco.type_model(t))
        log_D.append(np.log(eco.f[t]) + ld)
        S = S + eco.f[t] * s
    return logsumexp(jnp.stack(log_D), axis=0) - jnp.log(jnp.maximum(S, 1e-300))


def residual(z, eco):
    EVs, P = split_z(z, eco)
    bell = [EVs[t] - T_raw(EVs[t], P, eco.type_model(t)) for t in range(eco.n_types)]
    return jnp.concatenate(bell + [jnp.ravel(excess_demand_log(EVs, P, eco))])


@jax.jit
def _sa(EVs, P, eco):
    # 1,000 iteraciones de aproximaciones sucesivas por tipo: error ~ beta^1000
    out = [jax.lax.fori_loop(0, 1_000, lambda i, V, m=eco.type_model(t): T_raw(V, P, m), EVs[t])
           for t in range(eco.n_types)]
    return jnp.stack(out)


@jax.jit
def _newton(z, eco):
    F = residual(z, eco)
    Jz = jax.jacfwd(residual)(z, eco)
    return F, -jnp.linalg.solve(Jz, F)


@jax.jit
def _resid(z, eco):
    return residual(z, eco)


def _join(EVs, P):
    return jnp.concatenate([jnp.ravel(EVs), jnp.ravel(P)])


def initial_z(eco):
    J, A, n_act, n = dims(eco.g)
    P = initial_prices(eco.type_model(0))
    return _join(_sa(jnp.zeros((eco.n_types, n)), P, eco), P)


def solve(eco, z0=None, tol=1e-10, max_iter=50, verbose=False):
    # Newton con búsqueda de línea sobre ||F||.  Devuelve (z, convergió, iteraciones).
    z = initial_z(eco) if z0 is None else jnp.asarray(z0)
    for it in range(max_iter):
        F, dz = _newton(z, eco)
        err = float(jnp.max(jnp.abs(F)))
        if verbose:
            print(f"    Newton {it}: max|F| = {err:.2e}")
        if err < tol:
            return z, True, it
        nrm = float(jnp.linalg.norm(F))
        lam, ok = 1.0, False
        for _ in range(30):
            z_new = z + lam * dz
            F_new = _resid(z_new, eco)
            if bool(jnp.all(jnp.isfinite(F_new))) and float(jnp.linalg.norm(F_new)) < nrm:
                ok = True
                break
            lam *= 0.5
        if not ok:
            # Newton conjunto atorado: resolver los EV dados los P y seguir
            EVs, P = split_z(z, eco)
            z_new = _join(_sa(EVs, P, eco), P)
        z = z_new
    return z, False, max_iter


def solve_or_restart(eco, z0=None, **kw):
    # Arranque en caliente; si no converge, desde cero
    z, ok, it = solve(eco, z0, **kw)
    if not ok and z0 is not None:
        z, ok, it = solve(eco, None, **kw)
    return z, ok, it
