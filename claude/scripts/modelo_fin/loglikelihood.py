# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/loglikelihood.py
# Goal:           Verosimilitudes para recuperar la reparación no observada
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
Primera etapa de Hu & Xin con transición paramétrica (docs/id_modelo_vsHu.md, sec. 2).

Observaciones: tenencias h que sobreviven, con (t, j, d, s_h) -> s_next.
    usado h (d = 1..A-2):   f(s'|h, t) = (1 - p_t(h)) F_0(s'|h) + p_t(h) F_1(s'|h)
    nuevo h (d = 0):        f(s'|h)    = F_0(s'|h)            (los nuevos no se reparan)
    F_r(s'|h) = Pr(s' en la celda k | media m_r, sd),  m_r = c_j + s_age d + s_persist s - s_repair r
(misma discretización que el modelo: colas acumuladas en los extremos).

Tres estimadores:
- oráculo: observa r;  LL = sum log F_r(s').                       (cota superior)
- ingenuo: ignora la reparación; una sola F con s_repair = 0.     (cota inferior)
- hx:      r oculta; mezcla con pesos p_t(h).  Dos especificaciones de p:
    "flexible": logit(p) = alpha[t, j, d] + g1 s + g2 s^2   (libre por año: como Hu & Xin,
                que no le ponen forma a las CCPs; la exclusión entra porque F no cambia con t)
    "logit_R":  logit(p) = alpha_j + alpha_d + g1 s + g2 s^2 - b_R R_t(j, d)
s_repair = exp(.) > 0 fija el orden de los componentes (Assumption 3(iii) de Hu & Xin).
Sobrevivir no depende de r (opción 2), así que las salidas no aportan y se descartan.

Pendiente: segunda etapa estructural (mu, sigma_repair, ...).  Requiere que los
parámetros que se estiman sean dinámicos; hoy `g` es estático en jit y cada valor
nuevo recompila.  Ver docs/modelo_fin.md.
'''

import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.stats import norm
from scipy.optimize import minimize

from utils import make_s_grid


# Datos ______________________________________________________________

def treat_data(df, g, R=None, with_r=False):
    # Agrega a celdas (t, j, d, s, s', [r]) con conteos, como el cell_based del repo de Rust.
    A = g.a_max
    sel = ((df["tipo_h"] == "usado") & (df["d_h"] <= A - 2)) | (df["tipo_h"] == "nuevo")
    sel &= df["s_next_idx"] >= 0
    d = df.loc[sel].copy()
    d["d_h"] = np.where(d["tipo_h"] == "nuevo", 0, d["d_h"])
    keys = ["t", "j_h", "d_h", "s_h_idx", "s_next_idx"] + (["r"] if with_r else [])
    cell = d.groupby(keys).size().rename("cnt").reset_index()

    grid = np.asarray(make_s_grid(g))
    mid = 0.5 * (grid[1:] + grid[:-1])
    # Bordes finitos en vez de +-inf: en la derivada respecto a sd, +-inf da 0 * inf = nan.
    # 10 está a cientos de desviaciones estándar de cualquier s en [0, 1].
    lo = np.concatenate([[-10.0], mid])
    hi = np.concatenate([mid, [10.0]])
    k = cell["s_next_idx"].to_numpy()
    out = dict(
        t=jnp.asarray(cell["t"].to_numpy()),
        j=jnp.asarray(cell["j_h"].to_numpy()),
        d=jnp.asarray(cell["d_h"].to_numpy()),
        s=jnp.asarray(grid[cell["s_h_idx"].to_numpy()]),
        lo=jnp.asarray(lo[k]), hi=jnp.asarray(hi[k]),
        cnt=jnp.asarray(cell["cnt"].to_numpy(), dtype=float),
        new=jnp.asarray(cell["d_h"].to_numpy() == 0),
    )
    if with_r:
        out["r"] = jnp.asarray(cell["r"].to_numpy(), dtype=float)
    if R is not None:                                  # R_t(j, d), d = 1..A-1;  (T, J, A-1)
        dd = np.clip(cell["d_h"].to_numpy() - 1, 0, A - 2)
        out["R"] = jnp.asarray(R[cell["t"].to_numpy(), cell["j_h"].to_numpy(), dd])
    return out


# Piezas del modelo ______________________________________________________________

def cell_prob(mean, sd, lo, hi):
    # Pr(s' cae en [lo, hi)) con s' ~ N(mean, sd^2); log-seguro
    return jnp.clip(norm.cdf((hi - mean) / sd) - norm.cdf((lo - mean) / sd), 1e-300, 1.0)


def s_params(th):
    return dict(c=th["c"], s_age=th["s_age"], s_persist=th["s_persist"],
                s_repair=jnp.exp(th["log_srep"]), s_sigma=jnp.exp(th["log_sd"]))


def mean_r(th, data, r):
    sp = s_params(th)
    return sp["c"][data["j"]] + sp["s_age"] * data["d"] + sp["s_persist"] * data["s"] - sp["s_repair"] * r


def p_repair(th, data, spec):
    s = data["s"]
    base = th["g1"] * s + th["g2"] * s ** 2
    dd = jnp.clip(data["d"] - 1, 0, None)
    if spec == "flexible":
        idx = th["alpha"][data["t"], data["j"], dd]
    elif spec == "logit_R":
        idx = th["alpha_j"][data["j"]] + th["alpha_d"][dd] - th["b_R"] * data["R"]
    else:
        raise ValueError(spec)
    return jnp.where(data["new"], 0.0, jax.nn.sigmoid(idx + base))


# Verosimilitudes ______________________________________________________________

def ll_oracle(th, data):
    sd = jnp.exp(th["log_sd"])
    f = cell_prob(mean_r(th, data, data["r"]), sd, data["lo"], data["hi"])
    return jnp.sum(data["cnt"] * jnp.log(f))


def ll_naive(th, data):
    sd = jnp.exp(th["log_sd"])
    th0 = dict(th, log_srep=jnp.array(-jnp.inf))     # s_repair = 0
    f = cell_prob(mean_r(th0, data, 0.0), sd, data["lo"], data["hi"])
    return jnp.sum(data["cnt"] * jnp.log(f))


def ll_hx(th, data, spec):
    sd = jnp.exp(th["log_sd"])
    p = p_repair(th, data, spec)
    f0 = cell_prob(mean_r(th, data, 0.0), sd, data["lo"], data["hi"])
    f1 = cell_prob(mean_r(th, data, 1.0), sd, data["lo"], data["hi"])
    return jnp.sum(data["cnt"] * jnp.log((1.0 - p) * f0 + p * f1))


# Valores iniciales ______________________________________________________________

def start_values(g, kind, spec="flexible", n_regimes=None):
    J, A = g.n_brands, g.a_max
    th = dict(c=jnp.full(J, 0.02), s_age=jnp.array(0.005), s_persist=jnp.array(1.0),
              log_sd=jnp.log(jnp.array(0.02)))
    if kind in ("oracle", "hx"):
        th["log_srep"] = jnp.log(jnp.array(0.03))
    if kind == "hx":
        th["g1"], th["g2"] = jnp.array(0.0), jnp.array(0.0)
        if spec == "flexible":
            th["alpha"] = jnp.full((n_regimes, J, A - 2), -1.0)
        else:
            th["alpha_j"], th["alpha_d"], th["b_R"] = jnp.full(J, -1.0), jnp.full(A - 2, 0.0), jnp.array(0.0)
    return th


# Estimación ______________________________________________________________

def estim_ll(kind, data, th0, spec="flexible", se=False, verbose=False):
    # kind: "oracle" | "naive" | "hx".  Minimiza -LL con L-BFGS-B (como el repo de Rust).
    if kind == "naive":
        th0 = {k: v for k, v in th0.items() if k != "log_srep"}
    x0, unravel = ravel_pytree(th0)
    ll = {"oracle": lambda th: ll_oracle(th, data),
          "naive": lambda th: ll_naive(th, data),
          "hx": lambda th: ll_hx(th, data, spec)}[kind]
    N = float(jnp.sum(data["cnt"]))
    obj = jax.jit(jax.value_and_grad(lambda x: -ll(unravel(x)) / N))

    def fun(x):
        v, gr = obj(jnp.asarray(x))
        return float(v), np.asarray(gr)

    t0 = time.perf_counter()
    res = minimize(fun, np.asarray(x0), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=5_000, gtol=1e-9))
    th = unravel(jnp.asarray(res.x))
    if verbose:
        print(f"[{kind}] LL/N = {-res.fun:.6f}, it = {res.nit}, ok = {res.success}, "
              f"{time.perf_counter() - t0:.1f} s")

    out = dict(theta=th, ll=-res.fun * N, success=bool(res.success), nit=int(res.nit))
    if se:
        H = jax.hessian(lambda x: -ll(unravel(x)))(jnp.asarray(res.x))
        cov = jnp.linalg.inv(H)
        out["se"] = unravel(jnp.sqrt(jnp.clip(jnp.diag(cov), 0.0)))
    return out


def summarize_theta(th):
    # Parámetros de s en escala natural (para comparar con g)
    sp = s_params(th) if "log_srep" in th else dict(
        c=th["c"], s_age=th["s_age"], s_persist=th["s_persist"],
        s_repair=jnp.array(0.0), s_sigma=jnp.exp(th["log_sd"]))
    out = {f"s_const_{j}": float(v) for j, v in enumerate(np.asarray(sp["c"]))}
    out.update(s_age=float(sp["s_age"]), s_persist=float(sp["s_persist"]),
               s_repair=float(sp["s_repair"]), s_sigma=float(sp["s_sigma"]))
    return out


def true_theta(g):
    out = {f"s_const_{j}": v for j, v in enumerate(g.s_const)}
    out.update(s_age=g.s_age, s_persist=g.s_persist, s_repair=g.s_repair, s_sigma=g.s_sigma)
    return out
