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


def test_estructural():
    # Tamaño de juguete (calibración "tesis" con a_max = 7, n_s = 10, 2 regímenes, 3,000
    # hogares; log-odds y chatarreo endógeno): minutos.
    # 1) equilibrium.solve (Economy, 1 tipo) == ED.solve_equilibrium; dense == krylov con 2 tipos.
    # 2) gradiente implícito de la verosimilitud estructural contra diferencias finitas.
    import os, tempfile
    from params import Types
    from calibracion import calibracion
    from estimar import regimes_R, verdad
    from theta import economy, free_spec, pack, labels, FIELDS
    from equilibrium import solve, split_z, equilibrium_objects
    from primitives import s_transition
    from gen_dataset import simulate_economy
    from estructural import treat_data, LLEval, loglik_only

    g = dataclasses.replace(calibracion("tesis", a_max=7, n_s=10), ed_tol=1e-12, gmres_tol=1e-13)
    eco1 = economy(g, Types(("low_couple_poor",), (1.0,)))
    z1, ok1, _ = solve(eco1, method="dense")
    eq = solve_equilibrium(g)
    print("Economy 1 tipo == ED.py:", ok1, float(jnp.max(jnp.abs(split_z(z1, eco1)[1] - eq.P))))
    types = Types()
    eco = economy(g, types)
    zk, okk, _ = solve(eco, method="krylov")
    zd, okd, _ = solve(eco, method="dense")
    print("2 tipos, krylov == dense:", okk and okd, float(jnp.max(jnp.abs(zk - zd))))
    # sin chatarreo y sin cambiar nada más: el modelo de antes
    g0 = dataclasses.replace(g, scrap=False)
    z0, _, _ = solve(economy(g0, Types(("low_couple_poor",), (1.0,))), method="dense")
    eq0 = solve_equilibrium(dataclasses.replace(g0, mu=0.1131))
    print("scrap = False == ED.py sin chatarreo:", float(jnp.max(jnp.abs(split_z(z0, eco1)[1] - eq0.P))))

    th0 = eco.th
    spec = free_spec(th0, FIELDS)
    labs = labels(spec)
    zetas, Rs = regimes_R(g, 2, 0.3)
    zs = verdad(g, types, Rs, os.path.join(tempfile.gettempdir(), "tests_verdad.npz"), "dense")
    objs = [equilibrium_objects(z, eco.with_repair_price(R)) for z, R in zip(zs, Rs)]
    df = simulate_economy(objs, s_transition(g), g, types.f, 3000, 2, seed=0)
    print(f"panel: {len(df)} filas, {int(df.r.sum())} reparaciones, {int(df.chatarreo.sum())} chatarreos")
    x0 = np.asarray(pack(th0, spec)) * 1.02
    for info in ("oraculo", "hx"):
        data, _ = treat_data(df, g, info)
        ev = LLEval(g, types.f, spec, th0, Rs, data, info, zs0=zs, method="dense")
        ll, gr, _, _ = ev(x0)
        # error relativo a la escala del gradiente (componentes casi nulos inflan el error
        # componente a componente por redondeo de las diferencias finitas)
        h, fd = 1e-5, np.zeros_like(gr)
        f = lambda xx: float(loglik_only(ev.solve(xx), jnp.asarray(xx), th0, ev.Rs, data,
                                         ev.economies(x0)[0], spec, info))
        for k in range(len(x0)):
            e = np.zeros_like(x0)
            e[k] = h
            fd[k] = (f(x0 + e) - f(x0 - e)) / (2 * h)
        err = np.max(np.abs(gr - fd)) / np.max(np.abs(fd))
        print(f"gradiente implícito [{info}] vs dif. finitas: max|g - fd| / max|fd| = {err:.1e} "
              f"({len(labs)} parámetros)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed", action="store_true", help="también resolver el equilibrio")
    ap.add_argument("--n_s", type=int, default=None)
    ap.add_argument("--estructural", action="store_true", help="pruebas de equilibrium.py y estructural.py")
    args = ap.parse_args()

    g = Params() if args.n_s is None else dataclasses.replace(Params(), n_s=args.n_s)
    test_model(g)
    test_matrix_free(g)
    test_jacobians(g)
    if args.ed:
        test_equilibrium(g)
    if args.estructural:
        test_estructural()
