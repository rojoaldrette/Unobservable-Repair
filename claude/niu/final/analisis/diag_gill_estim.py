# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/final/analisis/diag_gill_estim.py
# Goal:           Diagnóstico de la no convergencia del Gillingham "parcial"
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           19/11/2026
#
# _____________________________________________________________________________
'''
    python -u diag_gill_estim.py --rep 1 [--info parcial] [--fix u1]

1. Estima Gillingham (como estimar_panel.py) con el optimizador en modo verboso.
2. LL en la verdad y en la estimación.
3. Dirección plana: eigen-descomposición de la información BHHH escalada (correlaciones) en
   θ̂; imprime los eigenvalores más chicos y qué parámetros cargan en ellos, y las
   correlaciones |ρ| > 0.9 entre estimadores.
4. Perfil de la LL a lo largo de la dirección más plana.
'''

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gillingham"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

from params import Config, theta                             # noqa: E402
from equilibrio import solve                                 # noqa: E402
from gen_dataset import to_cells                             # noqa: E402
from ll_estim import free_spec, pack, natural, labels, cell_data, LLEval, estimate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--info", default="parcial")
    ap.add_argument("--fix", default="", help="parámetros fijos en la verdad, separados por coma")
    ap.add_argument("--lbfgs_iter", type=int, default=500)
    ap.add_argument("--bhhh_iter", type=int, default=50)
    ap.add_argument("--propio", action="store_true", help="datos del propio modelo de Gillingham")
    args = ap.parse_args()
    cfg = Config()
    th0 = theta(cfg)
    spec = free_spec(th0, fix=tuple(f for f in args.fix.split(",") if f))
    labs = labels(spec)
    x0 = np.asarray(pack(th0, spec, cfg))
    truth = np.asarray(natural(x0, spec, th0, cfg))
    z0, ok, _ = solve(th0, cfg)
    if args.propio:
        # Datos generados por el propio modelo de Gillingham (sin w ni reparación), 30,000 hogares
        from equilibrio import equilibrium_objects
        from gen_dataset import simulate_panel
        df = simulate_panel(equilibrium_objects(z0, th0, cfg), cfg, 30_000, 4, seed=args.rep)
    else:
        df = pd.read_csv(os.path.join(HERE, "..", "output", "diag", f"panel_gill_rep{args.rep}.csv.gz"))
    cells = to_cells(df, args.info)
    ev = LLEval(cfg, spec, th0, cell_data(cells, args.info), args.info, z0=z0)
    print(f"{args.info}: {len(cells)} celdas, N = {ev.N:.0f}, {len(labs)} parámetros libres", flush=True)

    ll_true, g_true, _, _ = ev(x0)
    print(f"LL en la verdad: {ll_true:.3f}  |g|max/N = {np.max(np.abs(g_true)) / ev.N:.2e}", flush=True)

    t0 = time.time()
    r = estimate(ev, x0, args.lbfgs_iter, args.bhhh_iter, verbose=True)
    print(f"\nLL en θ̂: {r['ll']:.3f} (Δ vs verdad = {r['ll'] - ll_true:.3f}), convergió = {r['converged']}, "
          f"BHHH iters = {r['bhhh_iters']}, evals = {r['n_eval']}, fallas = {r['n_fail']}, "
          f"cond(B) = {r['cond_B']:.2e}, {time.time() - t0:.0f} s", flush=True)

    tab = pd.DataFrame(dict(parametro=labs, verdad=truth, estimado=r["theta"], se=r["se"]))
    print("\n" + tab.round(4).to_string(index=False))

    # Equilibrio de Gillingham en θ̂ contra el de la verdad: por marca y edad
    from equilibrio import equilibrium_objects
    from ll_estim import unpack
    from utils import split_states
    th_hat = unpack(np.asarray(r["x"]), spec, th0, cfg)
    oh, ot = equilibrium_objects(r["z"], th_hat, cfg), equilibrium_objects(z0, th0, cfg)
    rows = []
    for j in range(cfg.J):
        for a in range(1, cfg.a_max, 2):
            row = dict(j=j, a=a, P_hat=oh[0]["P"][j, a - 1], P_verdad=ot[0]["P"][j, a - 1])
            for nm, o in (("hat", oh), ("verdad", ot)):
                q = sum(f * split_states(x["q"], cfg)[0] for x, f in zip(o, cfg.f))
                keep = sum(f * split_states(x["q"], cfg)[0] * split_states(x["keep"], cfg)[0] for x, f in zip(o, cfg.f))
                scr = split_states(o[0]["scrap"], cfg)[0]
                row[f"q_{nm}"], row[f"keep_{nm}"], row[f"scrap_{nm}"] = q[j, a - 1], keep[j, a - 1] / q[j, a - 1], scr[j, a - 1]
            rows.append(row)
    print("\nGillingham por marca y edad: θ̂ contra verdad (scrap = Pr(chatarrear | se deshace), tipo 0)")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    np.save(os.path.join(HERE, "..", "output", "diag", f"x_hat_{args.info}{'_propio' if args.propio else ''}.npy"), r["x"])

    # Dirección plana en la escala de x (lo que ve el optimizador), con B escalada
    ll, g, B, _ = ev(r["x"])
    print(f"\nen θ̂: g'B^-1g/N = {float(g @ np.linalg.pinv(B) @ g) / ev.N:.2e}")
    d = np.sqrt(np.diag(B))
    Bs = B / np.outer(d, d)
    lam, V = np.linalg.eigh(Bs)
    print("\nEigenvalores más chicos de la información escalada (1 = sin correlación; ~0 = plano):")
    for k in range(4):
        load = sorted(zip(labs, V[:, k]), key=lambda t: -abs(t[1]))[:6]
        print(f"  λ = {lam[k]:.2e}: " + ", ".join(f"{n} {v:+.2f}" for n, v in load))
    cov = np.linalg.pinv(B)
    sd = np.sqrt(np.abs(np.diag(cov)))
    C = cov / np.outer(sd, sd)
    pares = [(labs[i], labs[j], C[i, j]) for i in range(len(labs)) for j in range(i) if abs(C[i, j]) > 0.9]
    print("\nCorrelaciones de los estimadores con |ρ| > 0.9:")
    for a, b, c in sorted(pares, key=lambda t: -abs(t[2]))[:15]:
        print(f"  {a:18s} {b:18s} {c:+.3f}")

    # Perfil a lo largo de la dirección más plana (en unidades de x, escaladas por d)
    v = V[:, 0] / d
    v = v / np.max(np.abs(v))
    print("\nPerfil de la LL en la dirección más plana (t = paso en el parámetro que más carga):")
    for t in (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0):
        lt = ev(r["x"] + t * v)[0]
        print(f"  t = {t:+.1f}: LL - LL(θ̂) = {lt - ll:+.4f}", flush=True)


if __name__ == "__main__":
    main()
