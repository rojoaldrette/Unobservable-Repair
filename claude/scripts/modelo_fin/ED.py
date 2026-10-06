# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/ED.py
# Goal:           Exceso de demanda ED(P) y solver del equilibrio estacionario
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
Equilibrio (Gillingham, sec. 3.4, adaptado a s y sin chatarreo):

    D(j,a,s) = trade_mass * Pr(comprar el usado (j,a,s) | trade)
    S(j,a,s) = q(j,a,s) * (1 - Pr(keep | j,a,s))          (todo lo que se suelta se vende)
    ED(P)    = 0  para todos los (j, a, s) con a = 1..A-1

Se resuelve en logs:  ED_log = log D - log max(S, ed_floor).
- log D se calcula analíticamente (log trade_mass + log buy, con logsumexp), así que
  nunca hay underflow y d log D / dP(h) ~ -mu/sigma_s: el jacobiano no es singular.
- Las celdas casi vacías (q < ed_floor; abajo de ~1e-13 q es ruido numérico) quedan con S = ed_floor y su
  precio se fija donde D = ed_floor.  Ese precio sale alto y no significa nada
  económico (nadie está ahí); se reporta qué celdas son "sin masa".
- P puede salir negativo: el modelo no tiene piso de precios (ver docs/inspeccion_zigzag.md).

Solver:
1. Tâtonnement con jacobiano diagonal: P <- P + damp * (sigma_s/mu) * ED_log.
2. Newton-Krylov: J v por jvp (sin armar el jacobiano de 1800 x 1800) y GMRES.
   Tampoco se arma M (n x n): q sale de `stationary_q` y los sistemas con I - beta M
   se resuelven con GMRES y `M_apply`.
   El efecto de P sobre EV entra por el teorema de la función implícita:
       (I - beta M) dEV = (dT/dP) dP.
'''

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from utils import dims, split_states, gmres
from bellman import T, choice_values, solve_bellman, log_buy_probs
from probabilities import ccps
from transitions import stationary_q, holding_distribution, holding_kernel, M_apply


# Componentes del mercado ______________________________________________________________

class Market(NamedTuple):
    # Todo lo que necesita ED(P), evaluado en (EV, P).  Formas "por estado":
    q: jnp.ndarray       # (n,)  distribución estacionaria sobre X (inicio de periodo)
    q_hold: jnp.ndarray  # (n,)  distribución de tenencias sobre H (= q Ω)
    keep: jnp.ndarray    # (J, A-1, S)  Pr(keep | x activo)
    trade: jnp.ndarray   # (J, A-1, S)  Pr(vender y comprar | x activo)
    purge: jnp.ndarray   # (J, A-1, S)  Pr(vender y quedarse sin coche | x activo)
    buy_used: jnp.ndarray  # (J, A-1, S)  Pr(comprar el usado h | trade)
    buy_new: jnp.ndarray   # (J,)         Pr(comprar el nuevo j | trade)
    repair: jnp.ndarray    # (J, A-1, S)  Pr(reparar | tener el usado h)
    trade_mass: jnp.ndarray  # ()  masa total que compra: q @ trade (incluye term y none)
    demand: jnp.ndarray    # (J, A-1, S)  D
    supply: jnp.ndarray    # (J, A-1, S)  S


def market_components(EV, P, g):
    J, A, S, n_act, n = dims(g)
    c = ccps(EV, P, g)
    q = stationary_q(c, g)
    pk, _, _ = split_states(c.keep, g)
    pt, _, _ = split_states(c.trade, g)
    pp, _, _ = split_states(c.purge, g)
    pr, _, _ = split_states(c.repair, g)
    qa, _, _ = split_states(q, g)
    buy_used = c.buy[:n_act].reshape(J, A - 1, S)
    trade_mass = q @ c.trade
    return Market(
        q=q,
        q_hold=holding_distribution(q, c),
        keep=pk, trade=pt, purge=pp,
        buy_used=buy_used,
        buy_new=c.buy[n_act:n_act + J],
        repair=pr,
        trade_mass=trade_mass,
        demand=trade_mass * buy_used,
        supply=qa * (1.0 - pk),
    )


# Exceso de demanda ______________________________________________________________

def excess_demand_log(EV, P, g):
    # ED_log(j, a, s) = log D - log max(S, floor).  Función pura de (EV, P): jvp-able.
    J, A, S, n_act, n = dims(g)
    c = ccps(EV, P, g)
    q = stationary_q(c, g)
    qa, _, _ = split_states(q, g)
    pk, _, _ = split_states(c.keep, g)

    v = choice_values(EV, P, g)
    log_buy_used = log_buy_probs(v.buy, g)[:n_act].reshape(J, A - 1, S)
    log_D = jnp.log(jnp.maximum(q @ c.trade, g.ed_floor)) + log_buy_used
    log_S = jnp.log(jnp.maximum(qa * (1.0 - pk), g.ed_floor))
    return log_D - log_S


ed_log_jit = jax.jit(excess_demand_log, static_argnames="g")


def excess_demand(P, g, EV_init=None):
    # ED_log(P) resolviendo antes la Bellman.  Devuelve (ED_log, EV).
    EV = solve_bellman(P, g, EV_init=EV_init)
    return ed_log_jit(EV, P, g), EV


# Jacobiano matrix-free ______________________________________________________________

def make_jvp(EV, P, g):
    # Devuelve v -> J_ED(P) v, con v del tamaño de P aplanado.  Sin matrices: el sistema
    # (I - beta M) dEV = dT/dP dP se resuelve con GMRES y productos M v.
    shp = P.shape
    c = ccps(EV, P, g)
    K = holding_kernel(c, g)
    I_bM = lambda v: v - g.beta * M_apply(v, c, K, g)

    T_P = lambda P_: T(EV, P_, g)
    ed = lambda EV_, P_: excess_demand_log(EV_, P_, g)

    @jax.jit
    def matvec(v):
        dP = v.reshape(shp)
        dT = jax.jvp(T_P, (P,), (dP,))[1]
        dEV = gmres(I_bM, dT, tol=g.gmres_tol, restart=g.gmres_restart,
                    maxiter=g.gmres_maxiter)            # (I - beta M) dEV = dT/dP dP
        return jax.jvp(ed, (EV, P), (dEV, dP))[1].ravel()

    return matvec


# Solver ______________________________________________________________

class Equilibrium(NamedTuple):
    P: jnp.ndarray
    EV: jnp.ndarray
    ed: jnp.ndarray          # ED_log en la solución
    empty: np.ndarray        # (J, A-1, S) bool: celdas sin masa (precio sin significado)
    converged: bool
    iters: tuple             # (tâtonnement, Newton)


def solve_equilibrium(g, P_init=None, EV_init=None, verbose=False):
    P = jnp.asarray(P_init) if P_init is not None else None
    if P is None:
        from utils import initial_prices
        P = initial_prices(g)
    step_scale = g.sigma_s / g.mu

    # 1. Tâtonnement con jacobiano diagonal
    ed, EV = excess_demand(P, g, EV_init)
    k_tat = 0
    for k_tat in range(g.tat_max_iter):
        err = float(jnp.max(jnp.abs(ed)))
        if verbose and k_tat % 10 == 0:
            print(f"  tât {k_tat}: max|ED_log| = {err:.2e}")
        if err < g.tat_tol:
            break
        P = P + g.tat_damp * step_scale * ed
        ed, EV = excess_demand(P, g, EV)

    # 2. Newton-Krylov con búsqueda lineal
    converged = False
    k_nk = 0
    for k_nk in range(g.nk_ed_max_iter):
        err = float(jnp.max(jnp.abs(ed)))
        if verbose:
            print(f"  NK-ED {k_nk}: max|ED_log| = {err:.2e}")
        if err < g.ed_tol:
            converged = True
            break
        matvec = make_jvp(EV, P, g)
        precond = lambda v: -step_scale * v              # inversa del jacobiano diagonal
        dP = gmres(matvec, -ed.ravel(), tol=g.gmres_tol, restart=g.gmres_restart,
                   maxiter=4 * g.gmres_maxiter, M=precond)    # el de P (7,200 incógnitas) necesita más ciclos
        dP = dP.reshape(P.shape)
        lam = 1.0
        for _ in range(12):
            ed_new, EV_new = excess_demand(P + lam * dP, g, EV)
            if float(jnp.max(jnp.abs(ed_new))) < err:
                break
            lam *= 0.5
        P, ed, EV = P + lam * dP, ed_new, EV_new

    m = market_components(EV, P, g)
    empty = np.asarray((m.supply < g.ed_mass_tol) & (m.demand < g.ed_mass_tol))
    if verbose:
        print(f"  equilibrio: convergió={converged}, max|ED_log|={float(jnp.max(jnp.abs(ed))):.2e}, "
              f"celdas sin masa={int(empty.sum())}/{empty.size}, "
              f"P<0 en celdas con masa={int(np.sum((np.asarray(P) < 0) & ~empty))}")
    return Equilibrium(P=P, EV=EV, ed=ed, empty=empty, converged=converged,
                       iters=(k_tat, k_nk))
