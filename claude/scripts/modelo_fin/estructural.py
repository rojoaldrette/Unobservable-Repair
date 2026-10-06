# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/estructural.py
# Goal:           Verosimilitud estructural (DNFXP) con r observada (D0) y no observada (D1)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           06/10/2026
#
# _____________________________________________________________________________
'''
Ver docs/verosimilitud_estructural.md.  Estimadores (`info`):

    "oraculo" (D0): se observa r.
    "hx"      (D1): r no se observa; se integra (mezcla de Hu & Xin con p = CCP del modelo).

Observación = un año de un hogar: (tipo τ, régimen t, x, o, h, [r], x').  Se agrega a
celdas con conteos.  Con c = CCPs del tipo τ en el equilibrio del régimen t:

    Pr(o, h | x) = keep(x)                 si o = keep   (h = x)
                 = purge(x)                si o = purge  (h = none; en none: seguir sin coche)
                 = trade(x) · buy(h)       si o = trade

    Pr(x' | h)  (h con coche; para h = none es 1):
       muere (x' = term):   p_term(h) = s(h)  (1 en la última edad)          [opción 2: no depende de r]
       sobrevive a s':     (1 − s(h)) · F(s' | h)
         D0:  F = Pr(r | h) F_r          y en la muerte también se multiplica Pr(r | h)
         D1:  F = (1 − p(h)) F_0 + p(h) F_1,   p(h) = c.repair(h)

    LL(θ) = sum_celdas cnt · [log Pr(o, h | x) + log Pr(x' | h)]

Precios no observados: P = P(θ) del equilibrio de cada régimen (DNFXP).  Cada régimen
(año, con su R_t) tiene su propio z_t = (EV_0, ..., EV_{T-1}, P).

Gradiente (función implícita, por régimen):  dz_t/dx = −F_z⁻¹ F_x.  Los scores por
celda salen de una jacfwd en x del log Pr evaluado en z_t + (dz_t/dx)(x' − x), igual que
en gillingham/loglikelihood.py.  De ahí: gradiente y matriz BHHH.

Optimización: L-BFGS (gradiente analítico) y luego BHHH (errores estándar).
`method` = "krylov" (GMRES, memoria O(n)) o "dense" (jacobiano denso, solo tamaño chico).
'''

import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from utils import dims, decode_states, make_s_grid, s_new_index
from primitives import s_transition
from probabilities import ccps_raw
from equilibrium import split_z, residual, solve_or_restart, solve_linear
from theta import Economy, unpack, natural, natural_jac_diag, labels

INFOS = ("oraculo", "hx")
O_CODE = {"keep": 0, "purge": 1, "stay_none": 1, "trade": 2}


# Datos ______________________________________________________________

def treat_data(df, g, info):
    # Panel de gen_dataset.simulate_economy -> celdas con conteos (arreglos jnp).
    # "hx" no usa la columna r (puede venir de view(df, "ideal"), que no la trae).
    J, A, S, n_act, n = dims(g)
    cols = ["tipo", "t", "x", "decision", "h", "x_next"] + (["r"] if info == "oraculo" else [])
    d = df[cols].copy()
    d["o"] = d["decision"].map(O_CODE)
    keys = ["tipo", "t", "x", "o", "h", "x_next"] + (["r"] if info == "oraculo" else [])
    cells = d.groupby(keys).size().rename("cnt").reset_index()
    if info != "oraculo":
        cells["r"] = 0
    kh, jh, dh, sh = decode_states(cells["h"].to_numpy(), g)
    kn, _, _, sn = decode_states(cells["x_next"].to_numpy(), g)
    i0 = s_new_index(g)
    data = dict(
        tau=cells["tipo"].to_numpy(), t=cells["t"].to_numpy(), x=cells["x"].to_numpy(),
        o=cells["o"].to_numpy(), h=cells["h"].to_numpy(), r=cells["r"].to_numpy(),
        cnt=cells["cnt"].to_numpy().astype(float),
        car=kh < 2,                                   # la tenencia es un coche (usado o nuevo)
        j_h=np.maximum(jh, 0), d_h=np.maximum(dh, 0),  # nuevo: d = 0
        s_h=np.where(kh == 0, sh, i0),                 # nuevo: s_new
        last=(kh == 0) & (dh == A - 1),                # usado en la última edad: muere seguro
        dead=kn == 1, s_next=np.maximum(sn, 0),
    )
    return {k: jnp.asarray(v) for k, v in data.items()}, cells


# Probabilidades por celda ______________________________________________________________

def _log(p):
    return jnp.log(jnp.clip(p, 1e-300, None))


def regime_ccps(z, eco):
    # (T_tipos, 5, n): keep, purge, trade sobre X; buy, repair sobre H
    EVs, P = split_z(z, eco)
    out = []
    for t in range(eco.n_types):
        c = ccps_raw(EVs[t], P, eco.type_model(t))
        out.append(jnp.stack([c.keep, c.purge, c.trade, c.buy, c.repair]))
    return jnp.stack(out)


def cell_logp(zs, ecos, data, info):
    # log Pr(celda) para todas las celdas.  zs[t], ecos[t]: equilibrio y economía del régimen t.
    C = jnp.stack([regime_ccps(zs[t], ecos[t]) for t in range(len(ecos))])   # (R, T, 5, n)
    tt, tau = data["t"], data["tau"]
    keep = C[tt, tau, 0, data["x"]]
    purge = C[tt, tau, 1, data["x"]]
    trade = C[tt, tau, 2, data["x"]]
    buy = C[tt, tau, 3, data["h"]]
    p = C[tt, tau, 4, data["h"]]
    o = data["o"]
    p_choice = jnp.where(o == 0, keep, jnp.where(o == 1, purge, trade * buy))

    m0 = ecos[0].type_model(0)                  # la dinámica de s es común a tipos y regímenes
    F = s_transition(m0)                        # (2, J, A, S, S)
    grid = make_s_grid(m0)
    idx = (data["j_h"], data["d_h"], data["s_h"], data["s_next"])
    F0, F1 = F[0][idx], F[1][idx]
    p_term = jnp.where(data["last"], 1.0, grid[data["s_h"]])
    if info == "oraculo":
        r = data["r"]
        p_r = jnp.where(r == 1, p, 1.0 - p)
        surv, die = p_r * jnp.where(r == 1, F1, F0), p_r
    else:
        surv, die = (1.0 - p) * F0 + p * F1, 1.0
    p_trans = jnp.where(data["car"],
                        jnp.where(data["dead"], p_term * die, (1.0 - p_term) * surv), 1.0)
    return _log(p_choice) + _log(p_trans)


# Verosimilitud, gradiente y BHHH ______________________________________________________________

def _ecos(x, spec, th_fixed, Rs, eco0):
    th = unpack(x, spec, th_fixed)
    return [Economy(eco0.g, dict(th, repair_price=Rs[t]), eco0.f) for t in range(Rs.shape[0])]


@partial(jax.jit, static_argnames=("spec", "info", "method"))
def score_parts(zs, x, th_fixed, Rs, data, eco0, spec, info, method):
    # eco0: Economy cuyo aux trae g y f (estáticos); sus hojas se ignoran
    ecos = _ecos(x, spec, th_fixed, Rs, eco0)
    Dz = []
    for t in range(Rs.shape[0]):
        Fx = jax.jacfwd(lambda x_: residual(zs[t], _ecos(x_, spec, th_fixed, Rs, eco0)[t]))(x)
        Dz.append(-solve_linear(zs[t], ecos[t], Fx, method))                  # dz_t/dx
    lp = cell_logp(zs, ecos, data, info)
    S = jax.jacfwd(lambda x_: cell_logp([zs[t] + Dz[t] @ (x_ - x) for t in range(len(Dz))],
                                        _ecos(x_, spec, th_fixed, Rs, eco0), data, info))(x)
    cnt = data["cnt"]
    return cnt @ lp, cnt @ S, S.T @ (cnt[:, None] * S)


@partial(jax.jit, static_argnames=("spec", "info"))
def loglik_only(zs, x, th_fixed, Rs, data, eco0, spec, info):
    # Solo la LL (para pruebas con diferencias finitas)
    return data["cnt"] @ cell_logp(zs, _ecos(x, spec, th_fixed, Rs, eco0), data, info)


class LLEval:
    # Evalúa LL(x) resolviendo los equilibrios de todos los regímenes, con arranque en caliente
    def __init__(self, g, f, spec, th_fixed, Rs, data, info, zs0=None, method="krylov"):
        self.g, self.f = g, tuple(f)
        self.spec, self.th_fixed, self.Rs = spec, th_fixed, jnp.asarray(Rs)
        self.data, self.info, self.method = data, info, method
        self.zs = None if zs0 is None else jnp.asarray(zs0)
        self.n_solves = 0

    def economies(self, x):
        th = unpack(jnp.asarray(x), self.spec, self.th_fixed)
        return [Economy(self.g, dict(th, repair_price=R), self.f) for R in self.Rs]

    def solve(self, x):
        zs = []
        for t, eco in enumerate(self.economies(x)):
            z0 = None if self.zs is None else self.zs[t]
            z, ok, _ = solve_or_restart(eco, z0, method=self.method)
            self.n_solves += 1
            if not ok:
                return None
            zs.append(z)
        return jnp.stack(zs)

    def __call__(self, x):
        zs = self.solve(x)
        if zs is None:
            return -np.inf, None, None, None
        eco0 = self.economies(x)[0]
        ll, gr, B = score_parts(zs, jnp.asarray(x), self.th_fixed, self.Rs, self.data, eco0,
                                self.spec, self.info, self.method)
        return float(ll), np.asarray(gr), np.asarray(B), zs


def estim_lbfgs(ev, x0, max_iter=500, verbose=False):
    # −LL/N con L-BFGS-B.  Si algún equilibrio no converge en un punto de prueba, valor
    # alto y la búsqueda de línea retrocede.
    t0 = time.perf_counter()
    N = float(jnp.sum(ev.data["cnt"]))

    def fun(x):
        ll, gr, _, zs = ev(x)
        if not np.isfinite(ll):
            return 1e10, np.zeros_like(x)
        ev.zs = zs
        return -ll / N, -gr / N

    res = minimize(fun, np.asarray(x0, float), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=max_iter, gtol=1e-7))
    if verbose:
        print(f"  L-BFGS: LL/N = {-res.fun:.8f}, it = {res.nit}, ok = {res.success}, "
              f"{time.perf_counter() - t0:.1f} s")
    return res.x


def estim_bhhh(ev, x0, max_iter=50, tol=1e-9, verbose=False):
    # BHHH con búsqueda de línea por mitades.  tol: g' B⁻¹ g / N (cambio esperado en LL/N)
    t0 = time.perf_counter()
    N = float(jnp.sum(ev.data["cnt"]))
    x = np.asarray(x0, float)
    ll, gr, B, zs = ev(x)
    if not np.isfinite(ll):
        raise RuntimeError("el equilibrio no converge en el valor inicial")
    ev.zs = zs
    converged, it, stall = False, 0, 0
    for it in range(max_iter):
        d = np.linalg.pinv(B) @ gr          # pinv: B singular = dirección no identificada
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
        stall = stall + 1 if ((out[0] - ll) / N < 1e-9 or lam < 1e-3) else 0
        x = x + lam * d
        ll, gr, B, zs = out
        ev.zs = zs
        if stall >= 3:
            break
    cov = np.linalg.pinv(B)
    sing = np.linalg.svd(B, compute_uv=False)
    Jd = np.asarray(natural_jac_diag(jnp.asarray(x), ev.spec, ev.th_fixed))
    nat = np.asarray(natural(jnp.asarray(x), ev.spec, ev.th_fixed))
    var = np.diag(cov)
    se = Jd * np.sqrt(np.abs(var))                             # delta method
    if sing[-1] < 1e-10 * sing[0]:                              # B casi singular: se no confiables
        se = np.where(var > 0, se, np.nan)
    if verbose:
        print(f"  BHHH: convergió = {converged}, it = {it}, equilibrios = {ev.n_solves}, "
              f"{time.perf_counter() - t0:.1f} s")
    return dict(x=x, theta=nat, se=se, cov_x=cov, ll=ll, zs=ev.zs, converged=converged,
                iters=it, labels=labels(ev.spec), cond_B=float(sing[0] / max(sing[-1], 1e-300)))


def estimate(ev, x0, lbfgs_iter=500, bhhh_iter=50, verbose=False):
    t0 = time.perf_counter()
    out = estim_bhhh(ev, estim_lbfgs(ev, x0, max_iter=lbfgs_iter, verbose=verbose),
                     max_iter=bhhh_iter, verbose=verbose)
    out["seconds"] = time.perf_counter() - t0
    return out
