# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/gillingham/tests.py
# Goal:           Pruebas rápidas del modelo y del estimador
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           07/10/2026
#
# _____________________________________________________________________________
'''
    python -u tests.py              # a_max = 10 (segundos)
    python -u tests.py --a_max 25   # tamaño del paper

1. Filas de Q, Ω y M suman 1; CCPs suman 1.
2. Equilibrio: Bellman y mercado en cero; flujo estacionario por marca (Teorema 3):
   coches que salen (chatarreo endógeno + terminal) = coches nuevos que entran.
3. dT/dEV = beta M (Lema L1), contra jacfwd.
4. Gradiente implícito de la LL (parcial y completa) contra diferencias finitas.
'''

import argparse

import numpy as np
import jax
import jax.numpy as jnp

from params import Config, theta
from utils import dims, split_states, type_theta
from bellman import T
from probabilidades import ccps
from equilibrio import solve, equilibrium_objects, split_z
from gen_dataset import simulate_panel, to_cells
from ll_estim import free_spec, pack, cell_data, score_parts, loglik, unpack


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FALLA'}] {name} {detail}")
    return ok


def main(a_max):
    cfg = Config(a_max=a_max)
    th = theta(cfg)
    J, A, n_act, n = dims(cfg)
    z, ok, it = solve(th, cfg)
    check("equilibrio converge", ok, f"({it} iteraciones de Newton)")
    objs = equilibrium_objects(z, th, cfg)

    for t, o in enumerate(objs):
        rows = o["keep"] + o["purge"] + o["trade"]
        check(f"tipo {t}: CCPs y buy suman 1", np.allclose(rows, 1) and np.isclose(o["buy"].sum(), 1))
        check(f"tipo {t}: filas de Q y M suman 1",
              np.allclose(o["Q"].sum(1), 1) and np.allclose(o["M"].sum(1), 1))
        check(f"tipo {t}: q = q M", np.allclose(o["q"] @ o["M"], o["q"], atol=1e-13) and o["q"].min() > 0)

    # Flujo estacionario (Teorema 3), agregado sobre tipos
    out, inflow = np.zeros(J), np.zeros(J)
    for o, f in zip(objs, cfg.f):
        qa, qt, _ = split_states(o["q"], cfg)
        ka, _, _ = split_states(o["keep"], cfg)
        sa, _, _ = split_states(o["scrap"], cfg)
        out += f * ((qa * (1 - ka) * sa).sum(1) + qt)
        inflow += f * (o["q"] @ o["trade"]) * o["buy"][n_act:n_act + J]
    check("flujo estacionario por marca", np.allclose(out, inflow, rtol=1e-8), f"{out} vs {inflow}")

    # Lema L1: dT/dEV = beta M
    EVs, P = split_z(z, cfg)
    tht = type_theta(th, 0)
    dT = jax.jacfwd(lambda V: T(V, P, tht, cfg))(EVs[0])
    check("dT/dEV = beta M", np.allclose(dT, cfg.beta * objs[0]["M"], atol=1e-12))

    # Gradiente implícito contra diferencias finitas centradas (con re-solución del equilibrio)
    df = simulate_panel(objs, cfg, 2_000, 4, seed=0)
    spec = free_spec(th)
    x0 = pack(th, spec, cfg)
    rng = np.random.default_rng(0)
    for info in ("parcial", "completa"):
        data = cell_data(to_cells(df, info), info)
        _, g, _ = score_parts(z, x0, th, data, cfg, spec, info)
        idx = rng.choice(len(x0), 6, replace=False)
        fd = []
        for i in idx:
            e = jnp.zeros(len(x0)).at[i].set(1e-5)
            lls = []
            for sgn in (1, -1):
                th_s = unpack(x0 + sgn * e, spec, th, cfg)
                z_s, ok_s, _ = solve(th_s, cfg, z0=z)
                lls.append(float(loglik(z_s, x0 + sgn * e, th, data, cfg, spec, info)))
            fd.append((lls[0] - lls[1]) / 2e-5)
        err = np.max(np.abs(np.asarray(g)[idx] - fd) / (1 + np.abs(fd)))
        check(f"gradiente implícito [{info}]", err < 1e-5, f"(error relativo {err:.1e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_max", type=int, default=10)
    main(ap.parse_args().a_max)
