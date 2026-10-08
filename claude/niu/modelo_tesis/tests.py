# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/modelo_tesis/tests.py
# Goal:           Pruebas rápidas del modelo teórico (hasta gen_dataset)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
    python -u tests.py          # a_max = 8, grid de w chico (segundos, CPU)

1. Transición de w: filas suman 1; media exacta en el interior (= w + delta - acc_age - kappa r).
2. Equilibrio del régimen central: converge; CCPs y buy suman 1.
3. q es estacionaria: un periodo completo (advance) la deja igual.
4. Flujo estacionario por marca: salen (chatarreo endógeno + terminal) = entran (nuevos).
5. Pr(reparar) = 0 donde no se permite (edad A-1); entre 0 y 1 en el resto.
6. Panel simulado: frecuencias de keep y de reparar cercanas a las del modelo.
'''

import numpy as np
import jax.numpy as jnp

from params import Config, theta, regime_theta
from utils import dims, split_states, type_theta
from bellman import w_transition
from probabilidades import ccps, advance
from equilibrio import solve, equilibrium_objects, split_z
from gen_dataset import simulate_panel


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FALLA'}] {name} {detail}")


def main():
    cfg = Config(a_max=8, w_step=0.15, w_min=-1.5, w_max=1.5, jac_chunk=128)
    th = regime_theta(theta(cfg), cfg, 1)
    J, A, W, n_act, n = dims(cfg)

    F = np.asarray(w_transition(type_theta(th, 0), cfg))
    g = cfg.w_grid
    mid = slice(W // 4, 3 * W // 4)
    m_true = g[None, mid] + np.asarray(th["delta"] - th["acc_age"])[:, None]
    m0 = (F[0] @ g)[:, mid]
    m1 = (F[1] @ g)[:, mid]
    check("transición de w: filas suman 1", np.allclose(F.sum(-1), 1))
    check("transición de w: media exacta en el interior",
          np.allclose(m0, m_true, atol=1e-6) and np.allclose(m1, m_true - float(th["kappa"]), atol=1e-6))

    z, ok, it = solve(th, cfg)
    check("equilibrio converge", ok, f"({it} iteraciones)")
    objs = equilibrium_objects(z, th, cfg)
    EVs, P = split_z(z, cfg)
    out, inflow = np.zeros(J), np.zeros(J)
    for t, (o, f) in enumerate(zip(objs, cfg.f)):
        check(f"tipo {t}: CCPs y buy suman 1",
              np.allclose(o["keep"] + o["purge"] + o["trade"], 1) and np.isclose(o["buy"].sum(), 1))
        c = ccps(EVs[t], P, type_theta(th, t), cfg)
        q2 = np.asarray(advance(jnp.asarray(o["q"]), c, type_theta(th, t), cfg))
        check(f"tipo {t}: q = q M", np.allclose(q2, o["q"], atol=1e-14), f"(error {np.abs(q2 - o['q']).max():.1e})")
        rep = o["repair"]
        check(f"tipo {t}: Pr(reparar) válida", np.all(rep[:, -1] == 0) and np.all((rep[:, :-1] > 0) & (rep[:, :-1] < 1)))
        qa, qt, _ = split_states(o["q"], cfg)
        ka, _, _ = split_states(o["keep"], cfg)
        sa, _, _ = split_states(o["scrap"], cfg)
        out += f * ((qa * (1 - ka) * sa).sum((1, 2)) + qt)
        inflow += f * (o["q"] @ o["trade"]) * o["buy"][n_act:n_act + J]
    check("flujo estacionario por marca", np.allclose(out, inflow, rtol=1e-8), f"{out} vs {inflow}")

    df = simulate_panel([objs], cfg, 20_000, 3, seed=1)
    d0 = df[df.tipo == 0]
    o = objs[0]
    keep_sim = (d0.o == 0).mean()
    keep_mod = float(o["q"] @ o["keep"])
    rep_rows = d0[d0.r >= 0]
    rep_mod = o["repair"].ravel()[rep_rows.h.to_numpy()].mean()
    check("panel: keep simulado ≈ modelo", abs(keep_sim - keep_mod) < 0.01, f"({keep_sim:.4f} vs {keep_mod:.4f})")
    check("panel: reparar simulado ≈ modelo", abs(rep_rows.r.mean() - rep_mod) < 0.01,
          f"({rep_rows.r.mean():.4f} vs {rep_mod:.4f})")


if __name__ == "__main__":
    main()
