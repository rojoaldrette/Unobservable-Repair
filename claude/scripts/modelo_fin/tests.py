# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/tests.py
# Goal:           Pruebas del modelo (correr: PYTHONIOENCODING=utf-8 python tests.py [--ed])
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________

import argparse
import dataclasses
import time

import numpy as np
import jax
import jax.numpy as jnp

from params import Params
from utils import dims, split_states, initial_prices, decode_states
from bellman import T, solve_bellman
from probabilities import ccps
from transitions import (physical_matrices, holding_transition, trade_matrices,
                         transition_matrix, stationary_distribution, holding_kernel,
                         holding_apply, holding_rapply, M_apply, M_rapply, stationary_q)
from ED import market_components, solve_equilibrium, make_jvp, excess_demand_log


def test_model(g):
    J, A, S, n_act, n = dims(g)
    P0 = initial_prices(g)

    t0 = time.time()
    EV = solve_bellman(P0, g, verbose=True)
    print(f"tiempo solver: {time.time() - t0:.1f} s")
    c = ccps(EV, P0, g)
    Om = trade_matrices(c, g)
    Q0, Q1 = physical_matrices(g)
    Q = holding_transition(c, g)
    M = transition_matrix(EV, P0, g)
    q = stationary_distribution(M)

    Om_tot = sum(Om.values())
    print("filas Omega suman 1:", float(jnp.max(jnp.abs(Om_tot.sum(1) - 1))))
    print("filas Q0, Q1, Q suman 1:", float(jnp.max(jnp.abs(Q0.sum(1) - 1))),
          float(jnp.max(jnp.abs(Q1.sum(1) - 1))), float(jnp.max(jnp.abs(Q.sum(1) - 1))))
    print("M rápida == M densa:", float(jnp.max(jnp.abs(M - Om_tot @ Q))))
    print("q >= 0, suma:", float(q.min()), float(q.sum()))
    print("q = q M:", float(jnp.max(jnp.abs(q @ M - q))))

    # decode_states es la inversa del layout
    kind, j, a, s = decode_states(np.arange(n), g)
    assert (kind[:n_act] == 0).all() and (kind[n_act:n_act + J] == 1).all() and kind[-1] == 2
    i = 1 * (A - 1) * S + 2 * S + 7                  # (j=1, a=3, s=7)
    assert (j[i], a[i], s[i]) == (1, 3, 7)
    print("decode_states OK")

    m = market_components(EV, P0, g)
    qa, qt, qn = split_states(m.q, g)
    print("sin coche:", float(qn), "  masa que compra:", float(m.trade_mass))
    return EV, P0


def test_matrix_free(g):
    # Las versiones sin matrices contra las densas (tamaño chico), y a_max = 25 sin densas.
    gs = dataclasses.replace(g, n_s=12)
    P = initial_prices(gs)
    EV = solve_bellman(P, gs)
    c = ccps(EV, P, gs)
    K = holding_kernel(c, gs)
    Q = holding_transition(c, gs)
    M = transition_matrix(EV, P, gs)
    v = jax.random.normal(jax.random.PRNGKey(1), EV.shape)
    print("Q v   sin matriz == densa:", float(jnp.max(jnp.abs(holding_apply(v, K, gs) - Q @ v))))
    print("v Q   sin matriz == densa:", float(jnp.max(jnp.abs(holding_rapply(v, K, gs) - v @ Q))))
    print("M v   sin matriz == densa:", float(jnp.max(jnp.abs(M_apply(v, c, K, gs) - M @ v))))
    print("v M   sin matriz == densa:", float(jnp.max(jnp.abs(M_rapply(v, c, K, gs) - v @ M))))
    q_d = stationary_distribution(M)
    q_s = stationary_q(c, gs)
    print("q recursión == q densa:", float(jnp.max(jnp.abs(q_s - q_d))),
          "  q = q M:", float(jnp.max(jnp.abs(M_rapply(q_s, c, K, gs) - q_s))))

    # a_max = 25: solo sin matrices (Bellman con NK-GMRES y q por recursión)
    g25 = dataclasses.replace(gs, a_max=25, repair_price=tuple(
        tuple(r[0] * (1 + 0.15 * a) for a in range(24)) for r in g.repair_price))
    P25 = initial_prices(g25)
    t0 = time.time()
    EV25 = solve_bellman(P25, g25, verbose=True)
    c25 = ccps(EV25, P25, g25)
    q25 = stationary_q(c25, g25)
    K25 = holding_kernel(c25, g25)
    print(f"a_max = 25, n_s = 12 (n = {EV25.shape[0]}): {time.time() - t0:.1f} s;  "
          f"sup|T(EV) - EV| = {float(jnp.max(jnp.abs(T(EV25, P25, g25) - EV25))):.1e};  "
          f"q = q M: {float(jnp.max(jnp.abs(M_rapply(q25, c25, K25, g25) - q25))):.1e}")


def test_jacobians(g):
    # Grid chico para que el autodiff denso sea barato
    gs = dataclasses.replace(g, n_s=12)
    P = initial_prices(gs)
    EV = solve_bellman(P, gs)
    Jac = jax.jacfwd(T)(EV, P, gs)
    print("dT/dEV == beta M:", float(jnp.max(jnp.abs(Jac - gs.beta * transition_matrix(EV, P, gs)))))

    # J_ED v (matrix-free) contra diferencias finitas resolviendo la Bellman
    matvec = make_jvp(EV, P, gs)
    v = jax.random.normal(jax.random.PRNGKey(0), P.shape)
    h = 1e-4                                         # el error de redondeo domina abajo de esto
    ed_p = excess_demand_log(solve_bellman(P + h * v, gs, EV), P + h * v, gs)
    ed_m = excess_demand_log(solve_bellman(P - h * v, gs, EV), P - h * v, gs)
    fd = ((ed_p - ed_m) / (2 * h)).ravel()
    jv = matvec(v.ravel())
    print("J_ED v (jvp) vs dif. finitas, error relativo:",
          float(jnp.max(jnp.abs(jv - fd)) / jnp.max(jnp.abs(fd))))


def test_equilibrium(g):
    t0 = time.time()
    eq = solve_equilibrium(g, verbose=True)
    print(f"tiempo equilibrio: {time.time() - t0:.1f} s")
    m = market_components(eq.EV, eq.P, g)
    full = ~eq.empty
    print("max |D - S| (celdas con masa):", float(np.max(np.abs(np.asarray(m.demand - m.supply))[full])))
    Pn = np.asarray(eq.P)
    print("P(j=0, a, s=s_new) por edad:", np.round(Pn[0, :, 6], 1))
    return eq


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed", action="store_true", help="también resolver el equilibrio")
    ap.add_argument("--n_s", type=int, default=None)
    args = ap.parse_args()

    g = Params() if args.n_s is None else dataclasses.replace(Params(), n_s=args.n_s)
    test_model(g)
    test_matrix_free(g)
    test_jacobians(g)
    if args.ed:
        test_equilibrium(g)
