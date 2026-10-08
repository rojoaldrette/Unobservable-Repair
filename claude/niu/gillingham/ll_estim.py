# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/gillingham/ll_estim.py
# Goal:           Máxima verosimilitud DNFXP (sec. 5.1 y apéndice D del paper)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           07/10/2026
#
# _____________________________________________________________________________
'''
Verosimilitud por celdas (ec. 40), condicional en la tenencia anterior h:

    L(θ) = sum_celdas cnt * log Pr(celda | θ, P(θ))

Con C[x, o] = Pr(resultado o | x) (probabilidades.outcome_probs), Q[h, x] la física y
buy(h') = Pr(comprar h' | trade):

    "parcial"  (apéndice D, ecs. 75-77; la del paper): no se ven los accidentes, así que
               x se integra:  Pr(o, h' | h) = (Q C)[h, o] * buy(h')^{1(o = trade)}
    "completa" (oráculo): se ve x:  Pr(x, o, h' | h) = Q[h, x] C[x, o] buy(h')^{1(o = trade)}

Los precios de usados no se observan: entra P(θ) del equilibrio, que se resuelve en cada
evaluación (el "doble" de DNFXP).  Tipos de hogar observados: cada celda usa lo de su tipo.

Gradiente analítico por la función implícita (z = (EV_0..EV_{T-1}, P), F de equilibrio.py):

    dz/dx = -F_z^{-1} F_x,     d log Pr(c)/dx = ∂_x + ∂_z dz/dx

Se calcula la matriz de scores por celda; de ahí el gradiente y la matriz BHHH.
Optimización: L-BFGS (gradiente analítico) para acercarse y BHHH (el del paper) para
terminar y sacar errores estándar.  BHHH solo, lejos del óptimo, avanza a pasos
minúsculos (el producto exterior de scores aproxima mal el hessiano ahí).

Parámetros libres (x, lo que ve el optimizador) <-> θ:
    mu = exp(x),  sigma_sell = sigma_trade * sigmoid(x)  (GEV válido),  resto = x.
Normalizaciones: sigma = sigma_trade = 1, u(sin coche) = 0.  Conocidos (fijos): p_new,
p_scrap, beta, u2 = 0.
'''

import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from params import TYPE_KEYS
from equilibrio import split_z, residual, per_type, solve_or_restart
from probabilidades import ccps, physical_matrix, outcome_probs

FREE = ("mu", "u0", "u1", "tc_buy", "tc_buy_nocar", "tc_sell", "tc_sell_inspect",
        "sigma_sell", "acc_int", "acc_age")


# Vector libre ______________________________________________________________

def free_spec(th, fix=()):
    # ((nombre, forma), ...): estático y hashable (entra a jit)
    return tuple((k, tuple(np.shape(th[k]))) for k in FREE if k not in fix)


def _fwd(k, v, cfg):
    if k == "mu":
        return jnp.exp(v)
    if k == "sigma_sell":
        return cfg.sigma_trade * jax.nn.sigmoid(v)
    return v


def _inv(k, v, cfg):
    if k == "mu":
        return jnp.log(v)
    if k == "sigma_sell":
        r = v / cfg.sigma_trade
        return jnp.log(r / (1 - r))
    return v


def pack(th, spec, cfg):
    return jnp.concatenate([jnp.ravel(_inv(k, th[k], cfg)) for k, _ in spec])


def unpack(x, spec, th_fixed, cfg):
    th, i = dict(th_fixed), 0
    for k, shape in spec:
        size = int(np.prod(shape))
        th[k] = _fwd(k, x[i:i + size].reshape(shape), cfg)
        i += size
    return th


def natural(x, spec, th_fixed, cfg):
    th = unpack(x, spec, th_fixed, cfg)
    return jnp.concatenate([jnp.ravel(th[k]) for k, _ in spec])


def labels(spec):
    # mu_t0, u0_t1_j0, tc_sell, acc_int_j1, ...
    out = []
    for k, shape in spec:
        for idx in np.ndindex(*shape):
            parts = [k]
            if k in TYPE_KEYS:
                parts.append(f"t{idx[0]}")
                idx = idx[1:]
            parts += [f"j{i}" for i in idx]
            out.append("_".join(parts))
    return out


# Datos y probabilidades por celda ______________________________________________________________

def cell_data(cells, info):
    col = lambda c: jnp.asarray(cells[c].to_numpy())
    d = dict(tau=col("tipo"), h_prev=col("h_prev"), o=col("o"), h=col("h"),
             trade=jnp.asarray(cells["o"].to_numpy() >= 3, dtype=float),
             cnt=jnp.asarray(cells["cnt"].to_numpy(), dtype=float))
    if info == "completa":
        d["x"] = col("x")
    return d


def _log(p):
    return jnp.log(jnp.clip(p, 1e-300, None))


def _type_probs(EV, P, th, cfg):
    c = ccps(EV, P, th, cfg)
    return physical_matrix(th, cfg), outcome_probs(c), c.buy


def cell_logp(z, th, cfg, data, info):
    EVs, P = split_z(z, cfg)
    Q, C, buy = per_type(_type_probs, EVs, P, th, cfg)          # (T, n, n), (T, n, 5), (T, n)
    tau = data["tau"]
    if info == "parcial":
        lp = _log(jnp.einsum("thx,txo->tho", Q, C)[tau, data["h_prev"], data["o"]])
    else:
        lp = _log(Q[tau, data["h_prev"], data["x"]]) + _log(C[tau, data["x"], data["o"]])
    return lp + data["trade"] * _log(buy[tau, data["h"]])


@partial(jax.jit, static_argnames=("cfg", "spec", "info"))
def score_parts(z, x, th_fixed, data, cfg, spec, info):
    # LL, gradiente y matriz BHHH en x, dado el equilibrio z(x)
    th_of = lambda x_: unpack(x_, spec, th_fixed, cfg)
    Fz = jax.jacfwd(residual)(z, th_of(x), cfg)
    Fx = jax.jacfwd(lambda x_: residual(z, th_of(x_), cfg))(x)
    Dz = -jnp.linalg.solve(Fz, Fx)                                         # dz/dx
    lp = cell_logp(z, th_of(x), cfg, data, info)
    S = jax.jacfwd(lambda x_: cell_logp(z + Dz @ (x_ - x), th_of(x_), cfg, data, info))(x)
    cnt = data["cnt"]
    return cnt @ lp, cnt @ S, S.T @ (cnt[:, None] * S)


@partial(jax.jit, static_argnames=("cfg", "spec", "info"))
def loglik(z, x, th_fixed, data, cfg, spec, info):
    # Solo la LL (pruebas con diferencias finitas)
    return data["cnt"] @ cell_logp(z, unpack(x, spec, th_fixed, cfg), cfg, data, info)


class LLEval:
    # LL(x) resolviendo el equilibrio con arranque en caliente desde el último z aceptado
    def __init__(self, cfg, spec, th_fixed, data, info, z0=None):
        self.cfg, self.spec, self.th_fixed, self.data, self.info = cfg, spec, th_fixed, data, info
        self.z = z0
        self.n_eval = self.n_fail = 0
        self.N = float(jnp.sum(data["cnt"]))

    def __call__(self, x):
        self.n_eval += 1
        th = unpack(jnp.asarray(x), self.spec, self.th_fixed, self.cfg)
        z, ok, _ = solve_or_restart(th, self.cfg, self.z)
        if not ok:
            self.n_fail += 1
            return -np.inf, None, None, None
        ll, g, B = score_parts(z, jnp.asarray(x), self.th_fixed, self.data, self.cfg,
                               self.spec, self.info)
        return float(ll), np.asarray(g), np.asarray(B), z


# Optimización ______________________________________________________________

def estim_lbfgs(ev, x0, max_iter, verbose=False):
    t0, st = time.perf_counter(), dict(it=0, f=np.nan, g=np.nan)

    def fun(x):
        ll, g, _, z = ev(x)
        if not np.isfinite(ll):
            return 1e10, np.zeros_like(x)            # la búsqueda de línea retrocede
        ev.z = z
        st["f"], st["g"] = -ll / ev.N, np.max(np.abs(g)) / ev.N
        return -ll / ev.N, -g / ev.N

    def log(xk):
        st["it"] += 1
        if verbose and st["it"] % 10 == 0:
            print(f"    L-BFGS {st['it']:4d}: LL/N = {-st['f']:.8f}, |g|max = {st['g']:.1e}, "
                  f"evals = {ev.n_eval}, {time.perf_counter() - t0:.0f} s", flush=True)

    res = minimize(fun, np.asarray(x0, float), jac=True, method="L-BFGS-B", callback=log,
                   options=dict(maxiter=max_iter, gtol=1e-6))
    if verbose:
        print(f"  L-BFGS: LL/N = {-res.fun:.8f}, it = {res.nit}, {res.message}, "
              f"{time.perf_counter() - t0:.1f} s", flush=True)
    return res.x


def estim_bhhh(ev, x0, max_iter, tol=1e-9, verbose=False):
    # BHHH con búsqueda de línea por mitades.  Para cuando g'B^{-1}g/N < tol (cambio esperado
    # en LL/N) o cuando la LL deja de subir 3 iteraciones seguidas.
    x = np.asarray(x0, float)
    ll, g, B, z = ev(x)
    if not np.isfinite(ll):
        raise RuntimeError("el equilibrio no converge en el valor inicial de BHHH")
    ev.z = z
    converged, stall, it = False, 0, 0
    for it in range(max_iter):
        d = np.linalg.pinv(B) @ g
        crit = float(g @ d) / ev.N
        if verbose:
            print(f"    BHHH {it:3d}: LL/N = {ll / ev.N:.8f}, g'B^-1g/N = {crit:.1e}", flush=True)
        if crit < tol:
            converged = True
            break
        lam = 1.0
        for _ in range(30):
            out = ev(x + lam * d)
            if np.isfinite(out[0]) and out[0] > ll:
                break
            lam *= 0.5
        else:
            break
        stall = stall + 1 if ((out[0] - ll) / ev.N < 1e-10 or lam < 1e-3) else 0
        x = x + lam * d
        ll, g, B, z = out
        ev.z = z
        if stall >= 3:
            break
    return x, ll, B, converged, it


def estimate(ev, x0, lbfgs_iter=500, bhhh_iter=50, verbose=False):
    # Devuelve θ̂ (escala natural), errores estándar (delta), LL, convergencia y diagnósticos
    t0 = time.perf_counter()
    x = estim_lbfgs(ev, x0, lbfgs_iter, verbose) if lbfgs_iter > 0 else np.asarray(x0, float)
    x, ll, B, converged, it = estim_bhhh(ev, x, bhhh_iter, verbose=verbose)
    cov = np.linalg.pinv(B)
    sv = np.linalg.svd(B, compute_uv=False)
    xj = jnp.asarray(x)
    jac = np.diag(np.asarray(jax.jacfwd(lambda v: natural(v, ev.spec, ev.th_fixed, ev.cfg))(xj)))
    var = np.diag(cov)
    se = np.where(var > 0, np.abs(jac) * np.sqrt(np.abs(var)), np.nan)
    return dict(x=x, theta=np.asarray(natural(xj, ev.spec, ev.th_fixed, ev.cfg)), se=se,
                ll=ll, z=ev.z, converged=converged, bhhh_iters=it, n_eval=ev.n_eval,
                n_fail=ev.n_fail, cond_B=float(sv[0] / max(sv[-1], 1e-300)),
                seconds=time.perf_counter() - t0)
