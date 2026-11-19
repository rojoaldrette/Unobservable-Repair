# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/final/analisis/diag_gill_panel.py
# Goal:           Regenerar el panel de una réplica en el espacio de estados de Gillingham
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           19/11/2026
#
# _____________________________________________________________________________
'''
    python -u diag_gill_panel.py --rep 1

Mismo panel que montecarlo.py (semilla = rep, N = 10,000, K = 4) con el equilibrio verdadero
de teoria/R_LB.  Lo guarda en output/diag/panel_gill_rep<rep>.csv.gz para diag_gill_estim.py.
'''

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "modelo_tesis"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from params import Config, theta, regime_theta                # noqa: E402
from equilibrio import equilibrium_objects                    # noqa: E402
from gen_dataset import simulate_panel                        # noqa: E402
from montecarlo import true_equilibria, to_gillingham         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--N", type=int, default=10_000)
    ap.add_argument("--K", type=int, default=4)
    args = ap.parse_args()
    cfg = Config()
    th = theta(cfg)
    zs = true_equilibria(th, cfg, "R_LB", verbose=False)
    objs = [equilibrium_objects(z, regime_theta(th, cfg, t), cfg) for t, z in enumerate(zs)]
    df = simulate_panel(objs, cfg, args.N, args.K, seed=args.rep)
    out = os.path.join(HERE, "..", "output", "diag")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"panel_gill_rep{args.rep}.csv.gz")
    to_gillingham(df, cfg).to_csv(path, index=False)
    df.to_csv(os.path.join(out, f"panel_rep{args.rep}.csv.gz"), index=False)
    print(f"listo: {path} ({len(df)} filas)")


if __name__ == "__main__":
    main()
