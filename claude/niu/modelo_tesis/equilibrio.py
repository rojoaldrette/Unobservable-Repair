# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/modelo_tesis/equilibrio.py
# Goal:           Equilibrio estacionario por régimen: z = (EV_0, EV_1, P), Newton denso
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
Para un régimen (th con su R_t), con tipos t de fracción f_t y precios comunes P(j, a, w):

    D(j, a, w) = sum_t f_t [sum_x q_t(x) trade_t(x)] buy_t(j, a, w)
    S(j, a, w) = sum_t f_t q_t(j, a, w) (1 - keep_t) (1 - scrap_t)
    F(z) = [ EV_t - Γ_t(EV_t, P)  (cada t) ;  log D - log S ] = 0

Newton con jacobiano denso (utils.jacobian, por bloques de cfg.jac_chunk columnas), búsqueda
de línea sobre ||F|| y tope al paso en precios.  Arranque: precios con depreciación de 13%
anual (iguales en w) y aproximaciones sucesivas para EV; entre regímenes, arranque en
caliente desde el régimen vecino (solve_regimes empieza por el central).

Celdas (j, a, w) con oferta casi nula (p. ej. w lejano a 0 en edades jóvenes): el piso
eps_F de la transición de w les da masa positiva pequeña y su precio es un precio sombra
alto que vuelve casi nula la demanda.  Las estadísticas se reportan ponderadas por masa.
'''

from functools import partial
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from params import regime_theta
from utils import dims, split_states, type_axes, type_theta, jacobian
from bellman import T, successive_approx, breakdown_prob, w_transition
from probabilidades import ccps, stationary


def split_z(z, cfg):
    J, A, W, n_act, n = dims(cfg)
    return z[:cfg.T * n].reshape(cfg.T, n), z[cfg.T * n:].reshape(J, A - 1, W)


def join_z(EVs, P):
    return jnp.concatenate([jnp.ravel(EVs), jnp.ravel(P)])


def per_type(fun, EVs, P, th, cfg):
    return jax.vmap(lambda EV, tht: fun(EV, P, tht, cfg), in_axes=(0, type_axes(th)))(EVs, th)


# Mercado ______________________________________________________________

def type_market(EV, P, th, cfg):
    # log demanda y oferta de un tipo (sin ponderar por f), (J, A-1, W)
    c = ccps(EV, P, th, cfg)
    q = stationary(c, th, cfg)
    qa, _, _ = split_states(q, cfg)
    keep, _, _ = split_states(c.keep, cfg)
    scrap, _, _ = split_states(c.scrap, cfg)
    return jnp.log(q @ c.trade) + c.lp_used, qa * (1 - keep) * (1 - scrap)


def excess_demand_log(EVs, P, th, cfg):
    log_D, S = per_type(type_market, EVs, P, th, cfg)
    f = jnp.asarray(cfg.f)
    log_D = logsumexp(log_D + jnp.log(f)[:, None, None, None], axis=0)
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
    F, Jz = jacobian(lambda v: residual(v, th, cfg), z, cfg.jac_chunk)
    return F, -jnp.linalg.solve(Jz, F)


@partial(jax.jit, static_argnames=("cfg", "n_iter"))
def _sa(EVs, P, th, cfg, n_iter):
    return per_type(lambda EV, P_, tht, c: successive_approx(EV, P_, tht, c, n_iter), EVs, P, th, cfg)


def initial_z(th, cfg):
    J, A, W, n_act, n = dims(cfg)
    a = jnp.arange(1, A)[None, :, None]
    P = jnp.maximum(th["p_new"][:, None, None] * 0.87 ** a, th["p_scrap"][:, None, None])
    P = jnp.broadcast_to(P, (J, A - 1, W))
    return join_z(_sa(jnp.zeros((cfg.T, n)), P, th, cfg, cfg.sa_iter), P)


def solve(th, cfg, z0=None, verbose=False, max_dP=150.0):
    # Newton con búsqueda de línea y tope max_dP (miles de DKK) al cambio de precios por paso.
    # Devuelve (z, convergió, iteraciones).
    J, A, W, n_act, n = dims(cfg)
    z = initial_z(th, cfg) if z0 is None else jnp.asarray(z0)
    t0 = time.time()
    for it in range(cfg.eq_max_iter):
        F, dz = _newton(z, th, cfg)
        err = float(jnp.max(jnp.abs(F)))
        if verbose:
            print(f"    Newton {it}: max|F| = {err:.2e}  ({time.time() - t0:.1f} s)", flush=True)
        if not np.isfinite(err):
            return z, False, it
        if err < cfg.eq_tol:
            return z, True, it
        big = float(jnp.max(jnp.abs(dz[cfg.T * n:])))
        dz = dz * min(1.0, max_dP / big) if big > 0 else dz
        nrm, lam = float(jnp.linalg.norm(F)), 1.0
        for _ in range(25):
            F_new = _resid(z + lam * dz, th, cfg)
            if bool(jnp.all(jnp.isfinite(F_new))) and float(jnp.linalg.norm(F_new)) < nrm:
                break
            lam *= 0.5
        else:
            EVs, P = split_z(z, cfg)               # Newton atorado: re-resolver EV dados P
            z = join_z(_sa(EVs, P, th, cfg, cfg.sa_iter), P)
            continue
        z = z + lam * dz
    return z, False, cfg.eq_max_iter


def solve_regimes(th, cfg, zs0=None, verbose=False):
    # Equilibrio de cada régimen.  Sin zs0: el central desde cero y los demás en caliente
    # desde su vecino hacia el centro.
    R = len(cfg.zetas)
    c = R // 2
    order = [c] + [t for k in range(1, R) for t in (c - k, c + k) if 0 <= t < R]
    zs, info = [None] * R, [None] * R
    for t in order:
        if zs0 is not None:
            z0 = zs0[t]
        elif t == c:
            z0 = None
        else:
            z0 = zs[t + 1 if t < c else t - 1]
        t0 = time.time()
        if verbose:
            print(f"  régimen {t} (zeta = {cfg.zetas[t]:+.2f})", flush=True)
        zs[t], ok, it = solve(regime_theta(th, cfg, t), cfg, z0, verbose)
        info[t] = dict(ok=ok, iters=it, segundos=time.time() - t0)
        if not ok:
            print(f"  OJO: el régimen {t} no convergió", flush=True)
    return zs, info


# Objetos de equilibrio y estadísticas ______________________________________________________________

def equilibrium_objects(z, th, cfg):
    # Por tipo (numpy): P, EV, q, CCPs, s (J, A, W), F (2, J, W, W)
    EVs, P = split_z(z, cfg)
    out = []
    for t in range(cfg.T):
        tht = type_theta(th, t)
        c = ccps(EVs[t], P, tht, cfg)
        d = dict(P=P, EV=EVs[t], q=stationary(c, tht, cfg), s=breakdown_prob(tht, cfg),
                 F=w_transition(tht, cfg), **c._asdict())
        out.append({k: np.asarray(v) for k, v in d.items()})
    return out


def market_stats(objs, cfg):
    # Agregados ponderados por f.  Tasas por coche en circulación (tenencias) al año.
    J, A, W, n_act, n = dims(cfg)
    st = dict(sin_coche=0.0, compra_nuevo=0.0, compra_usado=0.0, reparacion=0.0,
              chatarreo_endogeno=0.0, accidentes=0.0, edad_media=0.0, w_medio=0.0,
              masa_borde_w=0.0)
    for o, f in zip(objs, cfg.f):
        q = o["q"]
        m = q @ o["trade"]
        qa, _, qn = split_states(q, cfg)
        ka, _, _ = split_states(o["keep"], cfg)
        sa, _, _ = split_states(o["scrap"], cfg)
        hu = qa * ka + m * o["buy"][:n_act].reshape(J, A - 1, W)          # usados en mano
        hn = m * o["buy"][n_act:n_act + J]
        cars = hu.sum() + hn.sum()
        st["sin_coche"] += f * qn
        st["compra_nuevo"] += f * hn.sum()
        st["compra_usado"] += f * m * o["buy"][:n_act].sum()
        st["reparacion"] += f * (hu * o["repair"]).sum() / cars
        st["chatarreo_endogeno"] += f * (qa * (1 - ka) * sa).sum() / cars
        st["accidentes"] += f * ((hu[:, :A - 2] * o["s"][:, 1:A - 1]).sum()
                                 + (hn * o["s"][:, 0, cfg.w0]).sum()) / cars
        st["edad_media"] += f * (qa.sum((0, 2)) * np.arange(1, A)).sum() / qa.sum()
        st["w_medio"] += f * (qa.sum((0, 1)) * cfg.w_grid).sum() / qa.sum()
        st["masa_borde_w"] += f * (qa[:, :, [0, -1]].sum() / qa.sum())
    return {k: float(v) for k, v in st.items()}


def by_age(objs, cfg):
    # Por (j, a), promediado sobre w y tipos: q, precio medio (stock y transacciones),
    # keep, Pr(reparar) y w medio.  Lo que se compara con Gillingham.
    J, A, W, n_act, n = dims(cfg)
    P = objs[0]["P"]
    qa = sum(f * split_states(o["q"], cfg)[0] for o, f in zip(objs, cfg.f))
    S = sum(f * split_states(o["q"], cfg)[0] * (1 - split_states(o["keep"], cfg)[0])
            * (1 - split_states(o["scrap"], cfg)[0]) for o, f in zip(objs, cfg.f))
    kq = sum(f * split_states(o["q"], cfg)[0] * split_states(o["keep"], cfg)[0] for o, f in zip(objs, cfg.f))
    rq = sum(f * split_states(o["q"], cfg)[0] * o["repair"] for o, f in zip(objs, cfg.f))
    jj, aa = np.meshgrid(np.arange(J), np.arange(1, A), indexing="ij")
    qja = qa.sum(-1)
    return dict(j=jj.ravel(), a=aa.ravel(), q=qja.ravel(),
                P_stock=((qa * P).sum(-1) / qja).ravel(),
                P_transaccion=((S * P).sum(-1) / S.sum(-1)).ravel(),
                keep=(kq.sum(-1) / qja).ravel(), reparar=(rq.sum(-1) / qja).ravel(),
                w_medio=((qa * cfg.w_grid).sum(-1) / qja).ravel())
