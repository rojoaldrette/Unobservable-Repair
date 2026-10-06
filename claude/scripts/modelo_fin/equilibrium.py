# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/equilibrium.py
# Goal:           Equilibrio con T tipos de hogar y parámetros dinámicos (para estimar)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           06/10/2026
#
# _____________________________________________________________________________
'''
Para estimar hay que resolver el equilibrio en cada evaluación de la verosimilitud.
Este módulo lo hace con `theta.Economy` (parámetros como pytree: un θ nuevo no
recompila) y con varios tipos de hogar (fracciones f, precios P comunes):

    z = (EV_0, ..., EV_{T-1}, P)
    F(z, θ) = [ EV_t − T_t(EV_t, P)  para cada tipo t ;  ED_log(EV, P) ] = 0
    D(h) = sum_t f_t · trade_mass_t · Pr_t(comprar h | trade)
    S(x) = sum_t f_t · q_t(x) · (1 − keep_t(x)) · (1 − scrap_t(x))
    ED_log = log D − log max(S, ed_floor)

Newton sobre z con búsqueda de línea en ||F||.  La dirección se resuelve de dos formas
(`method`):
    "dense":  J_z = jacfwd(F) y un solve denso.  Exacto y simple; memoria O(n_z²): solo
              para tamaños chicos (pruebas) o GPU con memoria de sobra.
    "krylov": GMRES con productos J_z v por jvp (newton_krylov.md).  Memoria O(n_z):
              el que se usa con a_max = 25 y n_s = 100.
Precondicionador del bloque P: −sigma_s / mu medio (la diagonal aproximada de dED/dP).

Arranque (`initial_z`): Bellman en P0 y tâtonnement como en ED.py hasta tat_tol; luego
Newton.  En la estimación se arranca desde el z del θ anterior (`solve_or_restart`).

Con un tipo es el mismo equilibrio que ED.solve_equilibrium (se prueba en tests.py).
'''

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from utils import dims, split_states, initial_prices, gmres
from bellman import T_raw, choice_values, log_buy_probs
from probabilities import ccps_raw
from transitions import stationary_q, holding_kernel, M_apply


# z <-> (EVs, P) ______________________________________________________________

def split_z(z, eco):
    J, A, S, n_act, n = dims(eco.g)
    T = eco.n_types
    return z[:T * n].reshape(T, n), z[T * n:].reshape(J, A - 1, S)


def join_z(EVs, P):
    return jnp.concatenate([jnp.ravel(EVs), jnp.ravel(P)])


# Mercado ______________________________________________________________

def type_market(EV, P, m):
    # Para un tipo: log D_t (sin ponderar por f) y S_t (sin ponderar)
    J, A, S, n_act, n = dims(m)
    c = ccps_raw(EV, P, m)
    q = stationary_q(c, m)
    qa, _, _ = split_states(q, m)
    pk, _, _ = split_states(c.keep, m)
    psc, _, _ = split_states(c.scrap, m)
    v = choice_values(EV, P, m)
    log_buy = log_buy_probs(v.buy, m)[:n_act].reshape(J, A - 1, S)
    log_D = jnp.log(jnp.maximum(q @ c.trade, m.ed_floor)) + log_buy
    return log_D, qa * (1.0 - pk) * (1.0 - psc)          # lo chatarreado no se ofrece


def excess_demand_log(EVs, P, eco):
    log_D, S = [], 0.0
    for t in range(eco.n_types):
        ld, s = type_market(EVs[t], P, eco.type_model(t))
        log_D.append(np.log(eco.f[t]) + ld)
        S = S + eco.f[t] * s
    return logsumexp(jnp.stack(log_D), axis=0) - jnp.log(jnp.maximum(S, eco.g.ed_floor))


def residual(z, eco):
    EVs, P = split_z(z, eco)
    bell = [EVs[t] - T_raw(EVs[t], P, eco.type_model(t)) for t in range(eco.n_types)]
    return jnp.concatenate(bell + [jnp.ravel(excess_demand_log(EVs, P, eco))])


def _mu_bar(eco):
    return sum(f * eco.th["mu"][t] for t, f in enumerate(eco.f))


def _precond(eco):
    # Bloques EV: identidad (I − βM ya está bien condicionada).  Bloque P: −sigma_s/mu.
    J, A, S, n_act, n = dims(eco.g)
    nEV = eco.n_types * n
    scale = -eco.g.sigma_s / _mu_bar(eco)
    return lambda v: jnp.concatenate([v[:nEV], scale * v[nEV:]])


# Bellman por tipo, dado P ______________________________________________________________

def _bellman_type(EV, P, m, n_sa, n_nk):
    # n_sa iteraciones de T y luego n_nk pasos de Newton-Kantorovich (GMRES con M v)
    EV = jax.lax.fori_loop(0, n_sa, lambda i, V: T_raw(V, P, m), EV)

    def nk(i, V):
        c = ccps_raw(V, P, m)
        K = holding_kernel(c, m)
        r = V - T_raw(V, P, m)
        return V - gmres(lambda v: v - m.beta * M_apply(v, c, K, m), r, tol=m.gmres_tol,
                         restart=m.gmres_restart, maxiter=m.gmres_maxiter)

    return jax.lax.fori_loop(0, n_nk, nk, EV)


@partial(jax.jit, static_argnames=("n_sa", "n_nk"))
def bellman_types(EVs, P, eco, n_sa=300, n_nk=4):
    return jnp.stack([_bellman_type(EVs[t], P, eco.type_model(t), n_sa, n_nk)
                      for t in range(eco.n_types)])


@jax.jit
def _ed(EVs, P, eco):
    return excess_demand_log(EVs, P, eco)


@jax.jit
def _resid(z, eco):
    return residual(z, eco)


# Dirección de Newton ______________________________________________________________

@partial(jax.jit, static_argnames="method")
def newton_direction(z, eco, method="krylov"):
    F = residual(z, eco)
    if method == "dense":
        dz = -jnp.linalg.solve(jax.jacfwd(residual)(z, eco), F)
    else:
        g = eco.g
        Jv = lambda v: jax.jvp(lambda z_: residual(z_, eco), (z,), (v,))[1]
        dz = gmres(Jv, -F, tol=g.gmres_tol, restart=g.gmres_restart,
                   maxiter=4 * g.gmres_maxiter, M=_precond(eco))
    return F, dz


def solve_linear(z, eco, B, method="krylov"):
    # Resuelve F_z X = B (B: n_z x k).  Para el gradiente implícito: dz/dx = −F_z⁻¹ F_x.
    if method == "dense":
        return jnp.linalg.solve(jax.jacfwd(residual)(z, eco), B)
    g = eco.g
    Jv = lambda v: jax.jvp(lambda z_: residual(z_, eco), (z,), (v,))[1]
    one = lambda b: gmres(Jv, b, tol=g.gmres_tol, restart=g.gmres_restart,
                          maxiter=4 * g.gmres_maxiter, M=_precond(eco))
    return jax.vmap(one, in_axes=1, out_axes=1)(B)


# Solver ______________________________________________________________

def initial_z(eco, verbose=False):
    # Bellman en P0 y tâtonnement con jacobiano diagonal hasta tat_tol (como ED.py)
    g = eco.g
    J, A, S, n_act, n = dims(g)
    P = initial_prices(eco.type_model(0))
    EVs = bellman_types(jnp.zeros((eco.n_types, n)), P, eco)
    step = g.tat_damp * g.sigma_s / float(_mu_bar(eco))
    for k in range(g.tat_max_iter):
        ed = _ed(EVs, P, eco)
        err = float(jnp.max(jnp.abs(ed)))
        if verbose and k % 10 == 0:
            print(f"    tât {k}: max|ED_log| = {err:.2e}")
        if err < g.tat_tol:
            break
        P = P + step * ed
        EVs = bellman_types(EVs, P, eco, n_sa=20, n_nk=3)
    return join_z(EVs, P)


def solve(eco, z0=None, tol=None, max_iter=40, method="krylov", verbose=False):
    # Newton con búsqueda de línea sobre ||F||.  Devuelve (z, convergió, iteraciones).
    tol = eco.g.ed_tol if tol is None else tol
    z = initial_z(eco, verbose=verbose) if z0 is None else jnp.asarray(z0)
    for it in range(max_iter):
        F, dz = newton_direction(z, eco, method)
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
            # Newton atorado: resolver los EV dados los P y seguir
            EVs, P = split_z(z, eco)
            z_new = join_z(bellman_types(EVs, P, eco), P)
        z = z_new
    return z, False, max_iter


def solve_or_restart(eco, z0=None, **kw):
    # Arranque en caliente; si no converge, desde cero
    z, ok, it = solve(eco, z0, **kw)
    if not ok and z0 is not None:
        z, ok, it = solve(eco, None, **kw)
    return z, ok, it


# Objetos de equilibrio (numpy, para simular y reportar) ___________________________________

def equilibrium_objects(z, eco):
    # Lista por tipo con P, EV, q y CCPs; todo en numpy
    EVs, P = split_z(z, eco)
    out = []
    for t in range(eco.n_types):
        m = eco.type_model(t)
        c = ccps_raw(EVs[t], P, m)
        q = stationary_q(c, m)
        d = dict(P=P, EV=EVs[t], q=q, keep=c.keep, purge=c.purge, trade=c.trade, buy=c.buy,
                 repair=c.repair, scrap=c.scrap)
        out.append({k: np.asarray(v) for k, v in d.items()})
    return out
