# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         scripts/bellman.py
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
Quick comments:

1. 'g' es Params(), se controla desde main.  Es frozen + hashable, así que entra
   a jax.jit como argumento estático.  Los precios del mercado secundario P sí
   son argumento dinámico (el loop de equilibrio los va a mover).

2. Layout de estados (inicio de periodo, antes de comerciar), vector de largo n:
       X = [ act(j, a, s)  a=1..A-1  |  term(j)  |  none ]
   Layout de tenencias post-comercio (lo que manejas este periodo), mismo largo:
       H = [ used(j, d, s) d=1..A-1  |  new(j)   |  none ]
   (el truco de Gillingham: el hueco del terminal se recicla para el coche nuevo)
   n = J*(A-1)*S + J + 1   (3*6*100 + 3 + 1 = 1804 con los defaults)

3. Timing dentro del periodo:
   inicio en x -> elige {keep, repair, purge, trade} -> maneja h ->
   con prob s se descompone (-> term), si no s' = m(r,j,d,s) + eta, edad d+1.

4. Los shocks entran SOLO por las funciones `emax` y `choice_probs` y por los
   tres sigmas de Params.  Con sigmas iguales es logit simple.  Para otra
   distribución, cambia esas dos funciones (y el jacobiano de NK, ver nota ahí).
'''


# Packages and config ____________________________________________________________________

from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
from jaxopt import FixedPointIteration

from params_claude import Params

jax.config.update("jax_enable_x64", True)   # tolerancias de 1e-12 necesitan float64


# Grids and indexing ____________________________________________________________________

def dims(g):
    J, A, S = g.n_brands, g.a_max, g.n_s
    n_act = J * (A - 1) * S
    return J, A, S, n_act, n_act + J + 1


def make_s_grid(g):
    return jnp.linspace(g.s_min, g.s_max, g.n_s)


def s_new_index(g):
    # Python int (estático): índice del grid más cercano a s_new
    grid = np.linspace(g.s_min, g.s_max, g.n_s)
    return int(np.argmin(np.abs(grid - g.s_new)))


def split_states(x, g):
    # vector de largo n  ->  (act (J,A-1,S), term (J,), none ())
    J, A, S, n_act, n = dims(g)
    return x[:n_act].reshape(J, A - 1, S), x[n_act:n_act + J], x[-1]


def stack_states(act, term, none):
    return jnp.concatenate([jnp.ravel(act), jnp.ravel(term), jnp.reshape(none, (1,))])


# Primitives ____________________________________________________________________________

def flow_utility(g):
    # u(j, d, s) para d = 0..A-1 (d = 0 coche nuevo).  shape (J, A, S)
    d = jnp.arange(g.a_max)[None, :, None]
    s = make_s_grid(g)[None, None, :]
    u0 = jnp.asarray(g.u0)[:, None, None]
    u1 = jnp.asarray(g.u1)[:, None, None]
    u2 = jnp.asarray(g.u2)[:, None, None]
    return u0 + u1 * d + u2 * d ** 2 + g.u_s * s


def s_mean_next(g):
    # m(r, j, d, s): la "m(y, s)" de Hu & Xin.  shape (2, J, A, S)
    r = jnp.arange(2)[:, None, None, None]
    c = jnp.asarray(g.s_const)[None, :, None, None]
    d = jnp.arange(g.a_max)[None, None, :, None]
    s = make_s_grid(g)[None, None, None, :]
    return c + g.s_age * d + g.s_persist * s - g.s_repair * r


def s_transition(g):
    # F[r, j, d, s, s'] = Pr(s' | s, j, d, r, sobrevive).  shape (2, J, A, S, S)
    # Discretización por intervalos alrededor de cada punto del grid; las colas
    # se acumulan en los extremos (ojo: ahí E[eta|s] = 0 deja de cumplirse).
    grid = make_s_grid(g)
    mid = 0.5 * (grid[1:] + grid[:-1])
    edges = jnp.concatenate([jnp.array([-jnp.inf]), mid, jnp.array([jnp.inf])])
    m = s_mean_next(g)[..., None]
    cdf = norm.cdf((edges - m) / g.s_sigma)
    return jnp.diff(cdf, axis=-1)


# Shocks: lo único que cambia si cambias la distribución ___________________________________

def emax(values, sigma):
    # E max_k {v_k + sigma * eps_k}, eps EV1 iid (sin la constante de Euler,
    # como en Gillingham; solo desplaza EV en sigma*0.5772/(1-beta)).
    return sigma * logsumexp(jnp.stack(values, axis=0) / sigma, axis=0)


def choice_probs(values, sigma):
    # Probabilidades logit de cada alternativa (eje 0 = alternativa)
    v = jnp.stack(values, axis=0) / sigma
    return jnp.exp(v - logsumexp(v, axis=0, keepdims=True))


# Choice-specific values __________________________________________________________________

class Values(NamedTuple):
    keep: jnp.ndarray        # (J, A-1, S)
    repair: jnp.ndarray      # (J, A-1, S)
    purge_act: jnp.ndarray   # (J, A-1, S)   vender/chatarrear y quedarse sin coche
    trade_act: jnp.ndarray   # (J, A-1, S)   valor inclusivo del nido "trade"
    purge_term: jnp.ndarray  # (J,)
    trade_term: jnp.ndarray  # (J,)
    stay_none: jnp.ndarray   # ()
    trade_none: jnp.ndarray  # ()
    buy: jnp.ndarray         # (J*(A-1)*S + J,)  valor de comprar h, sin términos del vendedor
    disposal: jnp.ndarray    # (J, A-1, S)   valor inclusivo de vender vs chatarrear


def continuation_values(EV, g):
    # CV_r(h) = beta * E[EV(x') | h, r].
    # Devuelve cv_used (2, J, A-1, S) y cv_new (J,)  (el nuevo nunca se repara)
    J, A, S, n_act, n = dims(g)
    act, term, none = split_states(EV, g)
    F = s_transition(g)
    grid = make_s_grid(g)

    # usados de edad 1..A-2: sobreviven a edad d+1 con s' ~ F
    EV_next = jnp.einsum("rjdst,jdt->rjds", F[:, :, 1:A - 1], act[:, 1:])
    survive = (1.0 - grid) * EV_next + grid * term[None, :, None, None]
    # edad A-1: llega a terminal con certeza
    last = jnp.broadcast_to(term[None, :, None, None], (2, J, 1, S))
    cv_used = g.beta * jnp.concatenate([survive, last], axis=2)

    i0 = s_new_index(g)
    s0 = grid[i0]
    cv_new = g.beta * ((1.0 - s0) * jnp.sum(F[0, :, 0, i0, :] * act[:, 0, :], axis=-1)
                       + s0 * term)
    return cv_used, cv_new


def choice_values(EV, P, g):
    # P: precios del mercado secundario, shape (J, A-1, S), miles de DKK
    u = flow_utility(g)
    cv_used, cv_new = continuation_values(EV, g)
    act, term, none = split_states(EV, g)
    i0 = s_new_index(g)

    R = jnp.asarray(g.repair_price)[:, :, None]
    p_new = jnp.asarray(g.p_new)
    p_scrap = jnp.asarray(g.p_scrap)
    u_used = u[:, 1:, :]
    u_new = u[:, 0, i0]

    # Opciones de quien se queda con su coche
    keep = u_used + cv_used[0]
    repair = u_used - g.mu * R + cv_used[1]

    # Deshacerse del coche actual: vender (paga tc_sell) o chatarrear
    scrap_v = jnp.broadcast_to(g.mu * p_scrap[:, None, None], P.shape)
    sell_v = g.mu * P - g.tc_sell
    disposal = emax([scrap_v, sell_v], g.sigma_sell)

    # Comprar h: común a todos los estados (salvo una constante aditiva)
    buy_used = u_used - g.mu * P - g.tc_buy + cv_used[0]
    buy_new = u_new - g.mu * p_new - g.tc_buy + cv_new
    buy = jnp.concatenate([jnp.ravel(buy_used), buy_new])
    iv_buy = g.sigma_trade * logsumexp(buy / g.sigma_trade)   # logsum sobre TODAS las compras

    no_car_next = g.u_none + g.beta * none

    return Values(
        keep=keep,
        repair=repair,
        purge_act=disposal + no_car_next,
        trade_act=disposal + iv_buy,
        purge_term=g.mu * p_scrap + no_car_next,
        trade_term=g.mu * p_scrap + iv_buy,
        stay_none=no_car_next,
        trade_none=-g.tc_buy_nocar + iv_buy,
        buy=buy,
        disposal=disposal,
    )


# Wrappers  ______________________________________________________________________

# Operador de Bellman (Γ en Gillingham, ec. 9) sobre EV(x)
@partial(jax.jit, static_argnames="g")
def T(EV, P, g):
    v = choice_values(EV, P, g)
    ev_act = emax([v.keep, v.repair, v.purge_act, v.trade_act], g.sigma)
    ev_term = emax([v.purge_term, v.trade_term], g.sigma)
    ev_none = emax([v.stay_none, v.trade_none], g.sigma)
    return stack_states(ev_act, ev_term, ev_none)


# CCPs ______________________________________________________________________

class CCP(NamedTuple):
    keep: jnp.ndarray    # (n,) sobre X; 0 en term y none
    repair: jnp.ndarray  # (n,) sobre X; 0 en term y none
    purge: jnp.ndarray   # (n,) sobre X; en none = quedarse sin coche
    trade: jnp.ndarray   # (n,) sobre X; comprar algún coche
    buy: jnp.ndarray     # (n,) sobre H: Pr(comprar h | trade); 0 en la columna none
    scrap: jnp.ndarray   # (J, A-1, S): Pr(chatarrear | se deshace del coche), estática


@partial(jax.jit, static_argnames="g")
def ccps(EV, P, g):
    J, A, S, n_act, n = dims(g)
    v = choice_values(EV, P, g)

    p_act = choice_probs([v.keep, v.repair, v.purge_act, v.trade_act], g.sigma)
    p_term = choice_probs([v.purge_term, v.trade_term], g.sigma)
    p_none = choice_probs([v.stay_none, v.trade_none], g.sigma)

    zJ, z0 = jnp.zeros(J), jnp.zeros(())
    # Dentro del nido trade: la distribución de qué se compra NO depende de x
    # (separabilidad aditiva + logit).  Con nested logit sigue siendo así.
    p_buy = jnp.exp(v.buy / g.sigma_trade - logsumexp(v.buy / g.sigma_trade))

    scrap_v = jnp.broadcast_to(g.mu * jnp.asarray(g.p_scrap)[:, None, None], P.shape)
    p_scrap = choice_probs([scrap_v, g.mu * P - g.tc_sell], g.sigma_sell)[0]

    return CCP(
        keep=stack_states(p_act[0], zJ, z0),
        repair=stack_states(p_act[1], zJ, z0),
        purge=stack_states(p_act[2], p_term[0], p_none[0]),
        trade=stack_states(p_act[3], p_term[1], p_none[1]),
        buy=jnp.concatenate([p_buy, jnp.zeros(1)]),
        scrap=p_scrap,
    )


# Transition matrices ______________________________________________________________________

@partial(jax.jit, static_argnames="g")
def physical_matrices(g):
    # Q_r (n x n), de tenencia h (layout H) a estado x' (layout X).
    # Q_0: sin reparar; Q_1: reparado (solo difiere en filas de usados).
    J, A, S, n_act, n = dims(g)
    F = s_transition(g)
    grid = make_s_grid(g)
    i0 = s_new_index(g)
    s0 = grid[i0]
    eyeJ = jnp.eye(J)

    # usado (j,d,s) -> act (j,d+1,s'):  (1-s) F_r[j,d,s,s']   (shift mata d = A-1)
    shift = jnp.eye(A - 1, k=1)
    surv = (1.0 - grid)[None, None, None, :, None] * F[:, :, 1:]
    act_to_act = jnp.einsum("jk,de,rjdst->rjdsket", eyeJ, shift, surv).reshape(2, n_act, n_act)

    # usado -> term: prob s, o 1 si d = A-1
    is_last = (jnp.arange(A - 1) == A - 2)[:, None]
    p_term = jnp.where(is_last, 1.0, grid[None, :])                 # (A-1, S)
    act_to_term = (p_term[None, :, :, None] * eyeJ[:, None, None, :]).reshape(n_act, J)

    # nuevo j -> act (j, 1, s'), o term con prob s0
    first = (jnp.arange(A - 1) == 0).astype(grid.dtype)
    new_to_act = jnp.einsum("jk,e,jt->jket", eyeJ, first,
                            (1.0 - s0) * F[0, :, 0, i0, :]).reshape(J, n_act)
    new_to_term = s0 * eyeJ

    def assemble(r):
        top = jnp.concatenate([act_to_act[r], act_to_term, jnp.zeros((n_act, 1))], axis=1)
        mid = jnp.concatenate([new_to_act, new_to_term, jnp.zeros((J, 1))], axis=1)
        bot = jnp.concatenate([jnp.zeros((1, n_act + J)), jnp.ones((1, 1))], axis=1)
        return jnp.concatenate([top, mid, bot], axis=0)

    return assemble(0), assemble(1)


def trade_matrices(c, g):
    # Ω desagregada (n x n, de X a H).  Ω_total = suma de las cuatro; filas suman 1.
    J, A, S, n_act, n = dims(g)
    e_none = jnp.concatenate([jnp.zeros(n - 1), jnp.ones(1)])
    return dict(
        keep=jnp.diag(c.keep),                 # te quedas con el mismo coche, sin reparar
        repair=jnp.diag(c.repair),             # te quedas y reparas  (NO observable)
        trade=jnp.outer(c.trade, c.buy),       # vendes/chatarreas y compras h
        purge=jnp.outer(c.purge, e_none),      # te quedas sin coche
    )


@partial(jax.jit, static_argnames="g")
def transition_matrix(EV, P, g):
    # M = Ω_keep Q_0 + Ω_repair Q_1 + Ω_trade Q_0 + Ω_purge Q_0   (n x n, X -> X)
    # Se arma sin productos densos n^3: Ω_keep/repair son diagonales, Ω_trade es
    # de rango 1 y Ω_purge solo llena la columna none.
    c = ccps(EV, P, g)
    Q0, Q1 = physical_matrices(g)
    J, A, S, n_act, n = dims(g)
    e_none = jnp.concatenate([jnp.zeros(n - 1), jnp.ones(1)])
    return (c.keep[:, None] * Q0
            + c.repair[:, None] * Q1
            + c.trade[:, None] * (c.buy @ Q0)[None, :]
            + c.purge[:, None] * e_none[None, :])


def stationary_distribution(M):
    # q = q M,  sum(q) = 1   (Gillingham ec. 29, dado P)
    n = M.shape[0]
    A_ = jnp.concatenate([(M.T - jnp.eye(n))[:-1], jnp.ones((1, n))], axis=0)
    b = jnp.concatenate([jnp.zeros(n - 1), jnp.ones(1)])
    return jnp.linalg.solve(A_, b)


def observed_kept_transition(c, g):
    # Lo que ve el econometrista para coches que NO se comerciaron (Hu & Xin ec. 6.4):
    #   f(s'|j,a,s, kept) = [p_keep F_0 + p_repair F_1] / (p_keep + p_repair)
    # condicional en sobrevivir.  shape (J, A-1, S, S).  Útil para el Monte Carlo.
    J, A, S, n_act, n = dims(g)
    F = s_transition(g)[:, :, 1:]
    pk, _, _ = split_states(c.keep, g)
    pr, _, _ = split_states(c.repair, g)
    w = pr / (pk + pr)
    return (1.0 - w)[..., None] * F[0] + w[..., None] * F[1]


# Solver ______________________________________________________________________

def solve_bellman(P, g, EV_init=None, verbose=False):
    # Successive approximations (jaxopt) hasta sa_tol, luego Newton-Kantorovich.
    # Jacobiano de T: dT/dEV = beta * M (Gillingham, Lema L1).  Vale para cualquier
    # GEV porque el gradiente de emax son las CCPs; si cambias a una distribución
    # sin esa propiedad, usa jax.jacfwd(T) o un solver matrix-free.
    J, A, S, n_act, n = dims(g)
    EV = jnp.zeros(n) if EV_init is None else EV_init

    fpi = FixedPointIteration(fixed_point_fun=lambda V: T(V, P, g), implicit_diff=False,
                              maxiter=g.sa_max_iter, tol=g.sa_tol)
    sol = fpi.run(EV)
    EV = sol.params
    if verbose:
        print(f"SA: {int(sol.state.iter_num)} iteraciones, error {float(sol.state.error):.2e}")

    I = jnp.eye(n)
    for k in range(g.nk_max_iter):
        TEV = T(EV, P, g)
        err = float(jnp.max(jnp.abs(TEV - EV)))
        if verbose:
            print(f"NK {k}: sup|T(EV) - EV| = {err:.2e}")
        if err < g.vfi_tol:
            break
        M = transition_matrix(EV, P, g)
        EV = EV - jnp.linalg.solve(I - g.beta * M, EV - TEV)
    return EV


def initial_prices(g):
    # P0(j, a, s) = p_new_j * dep^a * (1 - s), acotado abajo por la chatarra
    a = jnp.arange(1, g.a_max)[None, :, None]
    s = make_s_grid(g)[None, None, :]
    p_new = jnp.asarray(g.p_new)[:, None, None]
    p_scrap = jnp.asarray(g.p_scrap)[:, None, None]
    return jnp.maximum(p_new * g.dep_factor ** a * (1.0 - s), p_scrap)


# Siguiente paso (tuyo): demanda y oferta agregadas ______________________________
#
#   c = ccps(EV, P, g);  q = stationary_distribution(transition_matrix(EV, P, g))
#   D(h)       = (q @ c.trade) * c.buy[h]                       h usado (primeros n_act)
#   S(j,a,s)   = q_act * (1 - p_keep - p_repair) * (1 - p_scrap)
#   ED(P)      = D - S   ->  resolver ED(P) = 0 (Newton sobre P, jax.jacfwd sirve aquí)


if __name__ == "__main__":
    # Pruebas
    g = Params()
    J, A, S, n_act, n = dims(g)
    P0 = initial_prices(g)

    EV = solve_bellman(P0, g, verbose=True)
    c = ccps(EV, P0, g)
    Om = trade_matrices(c, g)
    Q0, Q1 = physical_matrices(g)
    M = transition_matrix(EV, P0, g)
    q = stationary_distribution(M)

    Om_tot = sum(Om.values())
    print("filas Ω suman 1:", float(jnp.max(jnp.abs(Om_tot.sum(1) - 1))))
    print("filas Q suman 1:", float(jnp.max(jnp.abs(Q0.sum(1) - 1))),
          float(jnp.max(jnp.abs(Q1.sum(1) - 1))))
    M_dense = Om["keep"] @ Q0 + Om["repair"] @ Q1 + (Om["trade"] + Om["purge"]) @ Q0
    print("M rápida == M densa:", float(jnp.max(jnp.abs(M - M_dense))))
    print("q >= 0, suma:", float(q.min()), float(q.sum()))

    pk, _, _ = split_states(c.keep, g)
    pr, _, _ = split_states(c.repair, g)
    qa, qt, qn = split_states(q, g)
    print("sin coche:", float(qn))
    print("P(repair | j=0, a, s) en s = 0.05, 0.15, 0.30:")
    grid = np.linspace(g.s_min, g.s_max, g.n_s)
    for a in range(A - 1):
        idx = [int(np.argmin(abs(grid - x))) for x in (0.05, 0.15, 0.30)]
        print(f"  a={a + 1}:", np.round(np.asarray(pr[0, a, idx]), 3),
              " keep:", np.round(np.asarray(pk[0, a, idx]), 3))
