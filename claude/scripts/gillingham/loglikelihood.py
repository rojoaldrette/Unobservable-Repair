# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/loglikelihood.py
# Goal:           Máxima verosimilitud DNFXP (Gillingham sec. 5.1 y apéndice D)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           04/10/2026
#
# _____________________________________________________________________________
'''
Verosimilitud por celdas (ec. 40): L = sum_c cnt_c log Pr(c), condicional en la tenencia
anterior h.  Precios P no observados: entran los del equilibrio P(theta), resuelto en
cada evaluación (DNFXP).

Con C[x, o] = Pr(resultado o | x) (keep, purge/trade x vendido/chatarreado) y
Q[h, x] la transición física:

  "parcial"  (apéndice D, ecs. 75-77): no se ven los accidentes, así que x se integra:
             Pr(o, h' | h) = (Q C)[h, o] * buy(h')^{1(o es trade)}
             -> un coche que sale del parque puede ser accidente (x = terminal) o
                chatarreo endógeno (x activo, eligió chatarrear).
  "completa" (oráculo): se ve x, incluido el accidente:
             Pr(x, o, h' | h) = Q[h, x] C[x, o] buy(h')^{1(o es trade)}

Con tipos de hogar (observados), cada celda usa C, Q y buy de su tipo.

Gradiente: regla de la cadena a través del equilibrio,
    dz/dx = -F_z^{-1} F_x      (z = (EV_0..EV_{T-1}, P), F de equilibrium.py)
    d log Pr(c)/dx = parcial_x + parcial_z dz/dx
Se obtiene la matriz de scores por celda, así que de paso salen el gradiente y la
matriz BHHH (el paper usa BHHH con gradientes analíticos).

Optimización (`estimate`): L-BFGS con el gradiente analítico para acercarse al óptimo y
BHHH para terminar y sacar los errores estándar.  BHHH solo es rápido cerca del
óptimo: lejos, el producto exterior de scores aproxima mal al hessiano y avanza a
pasos minúsculos (visto en el Monte Carlo con arranques perturbados).
'''

import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from utils import dims, stack_states
from equilibrium import split_z, residual, solve_or_restart
from probabilities import ccps_raw
from transitions import physical_matrix_raw
from theta import Economy, unpack, natural, natural_jac_diag, labels


# Datos ______________________________________________________________

def treat_data(cells, info):
    # info = "parcial": se agrega sobre x (no se observa)
    if info == "parcial":
        cells = cells.groupby(["tipo", "h_prev", "o", "h"], as_index=False)["cnt"].sum()
        cells["x"] = 0
    o = cells["o"].to_numpy()
    return dict(tau=jnp.asarray(cells["tipo"].to_numpy()), h=jnp.asarray(cells["h_prev"].to_numpy()), x=jnp.asarray(cells["x"].to_numpy()),
                o=jnp.asarray(o), hn=jnp.asarray(cells["h"].to_numpy()),
                trade=jnp.asarray(o >= 3, dtype=float),
                cnt=jnp.asarray(cells["cnt"].to_numpy(), dtype=float))


# Probabilidades ______________________________________________________________

def outcome_probs(EV, P, m):
    J, A, n_act, n = dims(m)
    c = ccps_raw(EV, P, m)
    sx = stack_states(c.scrap, jnp.ones(J), jnp.zeros(()))
    C = jnp.stack([c.keep, c.purge * (1 - sx), c.purge * sx,
                   c.trade * (1 - sx), c.trade * sx], axis=1)          # (n, 5) sobre X
    return physical_matrix_raw(m), C, c.buy


def _log(p):
    return jnp.log(jnp.clip(p, 1e-300, None))


def cell_logp(z, eco, data, info):
    EVs, P = split_z(z, eco)
    Q, C, buy = map(jnp.stack, zip(*[outcome_probs(EVs[t], P, eco.type_model(t))
                                     for t in range(eco.n_types)]))   # (T, n, n), (T, n, 5), (T, n)
    tau = data["tau"]
    if info == "parcial":
        lp = _log(jnp.einsum("thx,txo->tho", Q, C)[tau, data["h"], data["o"]])
    else:
        lp = _log(Q[tau, data["h"], data["x"]]) + _log(C[tau, data["x"], data["o"]])
    return lp + data["trade"] * _log(buy[tau, data["hn"]])


# Verosimilitud, gradiente y BHHH ______________________________________________________________

@partial(jax.jit, static_argnames=("spec", "info"))
def _score_parts(z, x, th_fixed, data, eco0, spec, info):
    # eco0: Economy cuyo aux trae g y f (estáticos); sus hojas se ignoran
    mod = lambda x_: Economy(eco0.g, unpack(x_, spec, th_fixed), eco0.f)
    m = mod(x)
    Fz = jax.jacfwd(residual)(z, m)
    Fx = jax.jacfwd(lambda x_: residual(z, mod(x_)))(x)
    Dz = -jnp.linalg.solve(Fz, Fx)                                      # dz/dx
    lp = cell_logp(z, m, data, info)
    S = jax.jacfwd(lambda x_: cell_logp(z + Dz @ (x_ - x), mod(x_), data, info))(x)  # scores
    cnt = data["cnt"]
    return cnt @ lp, cnt @ S, S.T @ (cnt[:, None] * S)


class LLEval:
    # Evalúa LL(x) resolviendo el equilibrio con arranque en caliente
    def __init__(self, g, f, spec, th_fixed, data, info, z0=None):
        self.g, self.f = g, tuple(f)
        self.spec, self.th_fixed, self.data, self.info = spec, th_fixed, data, info
        self.z = z0
        self.n_solves = 0

    def model(self, x):
        return Economy(self.g, unpack(jnp.asarray(x), self.spec, self.th_fixed), self.f)

    def __call__(self, x):
        m = self.model(x)
        z, ok, _ = solve_or_restart(m, self.z)
        self.n_solves += 1
        if not ok:
            return -np.inf, None, None, z
        ll, gr, B = _score_parts(z, jnp.asarray(x), self.th_fixed, self.data, m,
                                 self.spec, self.info)
        return float(ll), np.asarray(gr), np.asarray(B), z


def estim_lbfgs(ev, x0, max_iter=500, verbose=False):
    # -LL/N con L-BFGS-B.  Si el equilibrio no converge en un punto de prueba, valor alto
    # y la búsqueda de línea retrocede.
    t0 = time.perf_counter()
    N = float(jnp.sum(ev.data["cnt"]))

    def fun(x):
        ll, gr, _, z = ev(x)
        if not np.isfinite(ll):
            return 1e10, np.zeros_like(x)
        ev.z = z
        return -ll / N, -gr / N

    res = minimize(fun, np.asarray(x0, float), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=max_iter, gtol=1e-7))
    if verbose:
        print(f"  L-BFGS: LL/N = {-res.fun:.8f}, it = {res.nit}, ok = {res.success}, "
              f"{time.perf_counter() - t0:.1f} s")
    return res.x


def estimate(ev, x0, verbose=False):
    t0 = time.perf_counter()
    out = estim_bhhh(ev, estim_lbfgs(ev, x0, verbose=verbose), max_iter=50, verbose=verbose)
    out["seconds"] = time.perf_counter() - t0
    return out


def estim_bhhh(ev, x0, max_iter=200, tol=1e-9, verbose=False):
    # BHHH con búsqueda de línea por mitades.  tol: g' B^{-1} g / N (cambio esperado en LL/N)
    t0 = time.perf_counter()
    N = float(jnp.sum(ev.data["cnt"]))
    x = np.asarray(x0, float)
    ll, gr, B, z = ev(x)
    if not np.isfinite(ll):
        raise RuntimeError("el equilibrio no converge en el valor inicial")
    ev.z = z
    converged, it, stall = False, 0, 0
    for it in range(max_iter):
        d = np.linalg.solve(B, gr)
        crit = float(gr @ d) / N
        if verbose:
            print(f"  BHHH {it:3d}: LL/N = {ll / N:.8f}, g'B^-1g/N = {crit:.2e}")
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
        # Estancado: la LL ya no sube (B casi singular o gradiente impreciso)
        stall = stall + 1 if ((out[0] - ll) / N < 1e-9 or lam < 1e-3) else 0
        x = x + lam * d
        ll, gr, B, z = out
        ev.z = z
        if stall >= 3:
            break
    cov = np.linalg.inv(B)
    J = np.asarray(natural_jac_diag(jnp.asarray(x), ev.spec, ev.th_fixed))
    nat = np.asarray(natural(jnp.asarray(x), ev.spec, ev.th_fixed))
    if verbose:
        print(f"  BHHH: convergió = {converged}, it = {it}, equilibrios = {ev.n_solves}, "
              f"{time.perf_counter() - t0:.1f} s")
    # Varianza negativa en la diagonal = B casi singular (dirección no identificada): nan
    var = np.diag(cov)
    se = np.where(var > 0, J * np.sqrt(np.abs(var)), np.nan)
    return dict(x=x, theta=nat, se=se, cov_x=cov, ll=ll, z=ev.z,
                converged=converged, iters=it, seconds=time.perf_counter() - t0,
                labels=labels(ev.spec))
