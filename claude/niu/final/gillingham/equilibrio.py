# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/gillingham/equilibrio.py
# Goal:           Equilibrio estacionario (P, q) con varios tipos de hogar
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           07/10/2026
#
# _____________________________________________________________________________
'''
Definición D1 y Teorema 4 del paper.  Con tipos t (fracciones f_t) y precios P comunes:

    D(j, a) = sum_t f_t * [sum_x q_t(x) trade_t(x)] * buy_t(j, a)          (ec. 21)
    S(j, a) = sum_t f_t * q_t(j, a) (1 - keep_t(j, a)) (1 - scrap_t(j, a))  (ec. 22)
    q_t = q_t M_t                                                         (ec. 34)

Se resuelve como un solo sistema en z = (EV_0, ..., EV_{T-1}, P):

    F(z, θ) = [ EV_t - Γ_t(EV_t, P)  (cada t) ;  log D(P) - log S(P) ] = 0

con Newton y jacobiano denso (T n + J (A-1) incógnitas: 150 con J = 2, T = 2, A = 25),
búsqueda de línea sobre ||F|| y arranque en caliente.  En logs: D y S van de 1e-6 a 1e-2
según la edad, y así todas las ecuaciones pesan parecido.

El paper anida dos Newton (Bellman dado P, luego P); resolver todo junto da el mismo
punto fijo y la derivada implícita dz/dθ = -F_z^{-1} F_θ sale del mismo jacobiano
(la usa ll_estim.py para el gradiente de la verosimilitud).
'''

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from utils import dims, split_states, type_axes, type_theta, initial_prices
from bellman import T, successive_approx, accident_prob
from probabilidades import (ccps, log_buy_probs, physical_matrix, transition_matrix,
                            stationary)


def split_z(z, cfg):
    J, A, n_act, n = dims(cfg)
    return z[:cfg.T * n].reshape(cfg.T, n), z[cfg.T * n:].reshape(J, A - 1)


def join_z(EVs, P):
    return jnp.concatenate([jnp.ravel(EVs), jnp.ravel(P)])


def per_type(fun, EVs, P, th, cfg):
    # fun(EV, P, th_t, cfg) evaluada para cada tipo (vmap sobre el eje de tipo)
    return jax.vmap(lambda EV, tht: fun(EV, P, tht, cfg), in_axes=(0, type_axes(th)))(EVs, th)


# Mercado ______________________________________________________________

def type_market(EV, P, th, cfg):
    # Un tipo: log de su demanda y su oferta de usados (sin ponderar por f), (J, A-1)
    c = ccps(EV, P, th, cfg)
    q = stationary(transition_matrix(c, physical_matrix(th, cfg)))
    qa, _, _ = split_states(q, cfg)
    keep, _, _ = split_states(c.keep, cfg)
    scrap, _, _ = split_states(c.scrap, cfg)
    log_D = jnp.log(q @ c.trade) + log_buy_probs(EV, P, th, cfg)
    return log_D, qa * (1 - keep) * (1 - scrap)


def excess_demand_log(EVs, P, th, cfg):
    log_D, S = per_type(type_market, EVs, P, th, cfg)            # (T, J, A-1)
    f = jnp.asarray(cfg.f)
    log_D = logsumexp(log_D + jnp.log(f)[:, None, None], axis=0)
    return log_D - jnp.log(jnp.maximum(jnp.tensordot(f, S, 1), 1e-300))


def residual(z, th, cfg):
    EVs, P = split_z(z, cfg)
    bell = EVs - per_type(T, EVs, P, th, cfg)
    return jnp.concatenate([jnp.ravel(bell), jnp.ravel(excess_demand_log(EVs, P, th, cfg))])


# Solver ______________________________________________________________

@partial(jax.jit, static_argnames="cfg")
def _resid(z, th, cfg):
    return residual(z, th, cfg)


@partial(jax.jit, static_argnames="cfg")
def _newton(z, th, cfg):
    F = residual(z, th, cfg)
    return F, -jnp.linalg.solve(jax.jacfwd(residual)(z, th, cfg), F)


@partial(jax.jit, static_argnames="cfg")
def _sa(EVs, P, th, cfg):
    return per_type(lambda EV, P_, tht, c: successive_approx(EV, P_, tht, c, cfg.sa_iter),
                    EVs, P, th, cfg)


def initial_z(th, cfg):
    # Precios con depreciación de 13% anual y EV por aproximaciones sucesivas dados esos P
    J, A, n_act, n = dims(cfg)
    P = initial_prices(th, cfg)
    return join_z(_sa(jnp.zeros((cfg.T, n)), P, th, cfg), P)


def solve(th, cfg, z0=None, verbose=False):
    # Newton con búsqueda de línea.  Devuelve (z, convergió, iteraciones).
    z = initial_z(th, cfg) if z0 is None else jnp.asarray(z0)
    for it in range(cfg.eq_max_iter):
        F, dz = _newton(z, th, cfg)
        err = float(jnp.max(jnp.abs(F)))
        if verbose:
            print(f"    Newton {it}: max|F| = {err:.2e}")
        if err < cfg.eq_tol:
            return z, True, it
        nrm, lam = float(jnp.linalg.norm(F)), 1.0
        for _ in range(30):
            F_new = _resid(z + lam * dz, th, cfg)
            if bool(jnp.all(jnp.isfinite(F_new))) and float(jnp.linalg.norm(F_new)) < nrm:
                break
            lam *= 0.5
        else:
            # Newton atorado (lejos de la solución): resolver los EV dados los P y seguir
            EVs, P = split_z(z, cfg)
            z = join_z(_sa(EVs, P, th, cfg), P)
            continue
        z = z + lam * dz
    return z, False, cfg.eq_max_iter


def solve_or_restart(th, cfg, z0=None):
    # Arranque en caliente; si no converge, desde cero
    z, ok, it = solve(th, cfg, z0)
    if not ok and z0 is not None:
        z, ok, it = solve(th, cfg)
    return z, ok, it


# Objetos de equilibrio y resumen ______________________________________________________________

def equilibrium_objects(z, th, cfg):
    # Lista por tipo (numpy): P, EV, q, CCPs, Q y M
    EVs, P = split_z(z, cfg)
    out = []
    for t in range(cfg.T):
        tht = type_theta(th, t)
        c = ccps(EVs[t], P, tht, cfg)
        Q = physical_matrix(tht, cfg)
        M = transition_matrix(c, Q)
        d = dict(P=P, EV=EVs[t], q=stationary(M), Q=Q, M=M, **c._asdict())
        out.append({k: np.asarray(v) for k, v in d.items()})
    return out


def market_stats(objs, th, cfg):
    # Estadísticas agregadas (ponderadas por f).  Tasas por coche en circulación al año.
    J, A, n_act, n = dims(cfg)
    al = np.asarray(accident_prob(type_theta(th, 0), cfg))
    st = dict(sin_coche=0.0, compra_nuevo=0.0, compra_usado=0.0, chatarreo_endogeno=0.0,
              accidentes=0.0, edad_media=0.0)
    for o, f in zip(objs, cfg.f):
        q = o["q"]
        mass_trade = q @ o["trade"]
        hold = o["keep"] * q + mass_trade * o["buy"]             # tenencia post-comercio sobre H
        hu, hn, _ = split_states(hold, cfg)
        cars = hu.sum() + hn.sum()
        qa, _, qn = split_states(q, cfg)
        ka, _, _ = split_states(o["keep"], cfg)
        sa, _, _ = split_states(o["scrap"], cfg)
        st["sin_coche"] += f * qn
        st["compra_nuevo"] += f * mass_trade * o["buy"][n_act:n_act + J].sum()
        st["compra_usado"] += f * mass_trade * o["buy"][:n_act].sum()
        st["chatarreo_endogeno"] += f * (qa * (1 - ka) * sa).sum() / cars
        st["accidentes"] += f * ((hu[:, :A - 2] * al[:, 1:A - 1]).sum() + (hn * al[:, 0]).sum()) / cars
        st["edad_media"] += f * (qa.sum(axis=0) * np.arange(1, A)).sum() / qa.sum()
    return {k: float(v) for k, v in st.items()}
