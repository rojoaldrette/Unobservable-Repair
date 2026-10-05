# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/tests.py
# Goal:           Pruebas de la réplica (PYTHONIOENCODING=utf-8 python tests.py [--a_max 7])
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

from params import GParams
from utils import dims, split_states, initial_prices
from bellman import T, solve_bellman
from probabilities import ccps
from transitions import physical_matrix, transition_matrix, stationary_distribution
from ED import solve_equilibrium, market_components


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_max", type=int, default=25)
    args = ap.parse_args()
    g = dataclasses.replace(GParams(), a_max=args.a_max)
    J, A, n_act, n = dims(g)

    P0 = initial_prices(g)
    EV = solve_bellman(P0, g)
    Q = physical_matrix(g)
    M = transition_matrix(EV, P0, g)
    print("filas Q, M suman 1:", float(jnp.max(jnp.abs(Q.sum(1) - 1))), float(jnp.max(jnp.abs(M.sum(1) - 1))))
    Jac = jax.jacfwd(T)(EV, P0, g)
    print("dT/dEV == beta M:", float(jnp.max(jnp.abs(Jac - g.beta * M))))

    t0 = time.time()
    eq = solve_equilibrium(g, verbose=True)
    print(f"tiempo: {time.time() - t0:.1f} s")
    m = market_components(eq.EV, eq.P, g)
    qa, qt, qn = split_states(m.q, g)
    print("max |D - S|:", float(jnp.max(jnp.abs(m.demand - m.supply))))
    print(f"sin coche: {float(qn):.3f}")
    print("P (j=0) por edad:", np.round(np.asarray(eq.P)[0], 1))
    print("chatarreo endógeno (j=0) por edad:", np.round(np.asarray(m.endo_scrap)[0], 3))
    print("accidente (j=0) por edad:", np.round(np.asarray(m.accident)[0, 1:], 3))

    # Estimación (equilibrium.py, loglikelihood.py) ______________________________
    from theta import model, theta_from_g, free_spec, pack
    from equilibrium import solve, split_z
    from simulate import equilibrium_objects, simulate_panel, to_cells
    from loglikelihood import treat_data, LLEval

    mod = model(g)
    z, ok, _ = solve(mod)
    EV_f, P_f = split_z(z, g)
    print("solver conjunto == solver de ED.py:", float(jnp.max(jnp.abs(P_f - eq.P))),
          float(jnp.max(jnp.abs(EV_f - eq.EV))))

    cells = to_cells(simulate_panel(equilibrium_objects(z, mod), g, 5_000, 5, seed=0))
    th0 = theta_from_g(g)
    spec = free_spec(g)
    x0 = np.asarray(pack(th0, spec))
    for info in ("parcial", "completa"):
        ev = LLEval(g, spec, th0, treat_data(cells, info), info, z0=z)
        ll, gr, B, _ = ev(x0)
        eps, fd = 1e-5, []
        for i in range(len(x0)):
            e = np.zeros(len(x0)); e[i] = eps
            fd.append((ev(x0 + e)[0] - ev(x0 - e)[0]) / (2 * eps))
        print(f"gradiente implícito vs diferencias finitas ({info}):",
              float(np.max(np.abs(gr - np.array(fd))) / np.max(np.abs(gr))))
