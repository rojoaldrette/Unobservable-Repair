# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/modelo_tesis/teoria.py
# Goal:           Resultados del modelo teórico (sin estimar): equilibrios, comparación con
#                 Gillingham y un panel simulado
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
Uso (desde claude/niu/modelo_tesis/):

    python -u teoria.py -v                         # tamaño completo (GPU): a_max 25, h = 0.075
    python -u teoria.py --prueba -v                # chico, para revisar que todo corre (CPU)
    python -u teoria.py --u_w 0 --tag sin_uw -v    # cambiar un parámetro desde la línea de comandos

Qué hace:
1. Resuelve el equilibrio de los 3 regímenes de R (el central desde cero, los demás en caliente).
2. Resuelve Gillingham con las mismas marcas, tipos y a_max (niu/gillingham/exportar.py,
   en un proceso aparte).
3. Compara por (marca, edad), promediando sobre w: precio medio (ponderado por el stock y por
   lo que se ofrece), distribución q(j, a), keep, Pr(reparar) y w medio.
4. Simula el panel de gen_dataset (N hogares por régimen, K años) y resume lo observado.

Salidas en claude/niu/output/modelo_tesis/teoria/<tag>/:
    config.json          Config y θ
    equilibrios.npz      z de cada régimen
    resumen.csv          una fila por régimen (y una de Gillingham): agregados de mercado y tiempos
    por_edad.csv         regimen, j, a, q, P_stock, P_transaccion, keep, reparar, w_medio,
                         q_gill, P_gill, keep_gill
    panel_resumen.csv    del panel simulado, por régimen: tasas observadas
    panel.csv.gz         (con --guardar_panel) el panel completo
'''

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import pandas as pd
import jax

from params import Config, theta, regime_theta
from utils import dims
from equilibrio import solve_regimes, equilibrium_objects, market_stats, by_age
from gen_dataset import simulate_panel

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.normpath(os.path.join(HERE, "..", "output", "modelo_tesis", "teoria"))


def gillingham(cfg, outdir):
    # Corre niu/gillingham/exportar.py en un proceso aparte y lee sus salidas
    path = os.path.join(outdir, "gill_por_edad.csv")
    if not os.path.exists(path):
        subprocess.run([sys.executable, "-u", "exportar.py", "--outdir", outdir,
                        "--a_max", str(cfg.a_max), "--brands", ",".join(cfg.brands),
                        "--types", ",".join(cfg.types), "--nocar", cfg.nocar],
                       cwd=os.path.join(HERE, "..", "gillingham"), check=True)
    with open(os.path.join(outdir, "gill_resumen.json"), encoding="utf-8") as fh:
        return pd.read_csv(path), json.load(fh)


def panel_resumen(df, cfg):
    # Lo observado en el panel simulado, por régimen
    J, A, W, n_act, n = dims(cfg)
    car = df["h"] < n - 1
    rep = df["r"] >= 0
    g = df.assign(car=car, rep_ok=rep, r1=df["r"] == 1).groupby("regimen")
    return pd.DataFrame(dict(
        hogar_anios=g.size(),
        sin_coche=g.apply(lambda d: (d["h"] == n - 1).mean()),
        tasa_reparacion=g.apply(lambda d: d.loc[d["rep_ok"], "r1"].mean()),
        accidentes_por_coche=g.apply(lambda d: d.loc[d["car"], "accidente"].mean()),
        chatarreo_endogeno=g.apply(lambda d: d["chatarreo_endogeno"].sum() / d["car"].sum()),
        compra=g.apply(lambda d: (d["o"] >= 3).mean()),
    )).reset_index()


def tabla(df, gill, regimen, ages=(1, 2, 4, 5, 10, 15, 20, 24)):
    # Impresión compacta: régimen central contra Gillingham
    d = df[(df.regimen == regimen) & df.a.isin(ages)]
    cols = ["j", "a", "P_stock", "P_gill", "q", "q_gill", "keep", "keep_gill", "reparar", "w_medio"]
    print(d[cols].round(3).to_string(index=False))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_max", type=int, default=Config.a_max)
    ap.add_argument("--w_step", type=float, default=Config.w_step)
    ap.add_argument("--w_min", type=float, default=Config.w_min)
    ap.add_argument("--w_max", type=float, default=Config.w_max)
    ap.add_argument("--jac_chunk", type=int, default=Config.jac_chunk)
    ap.add_argument("--u_w", type=float, default=None, help="reemplaza u_w (p. ej. 0)")
    ap.add_argument("--N", type=int, default=10_000, help="hogares por régimen en el panel")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--guardar_panel", action="store_true")
    ap.add_argument("--prueba", action="store_true", help="tamaño chico para CPU")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse()
    if args.prueba:
        args.a_max, args.w_step, args.w_min, args.w_max = 10, 0.15, -1.5, 1.5
        args.N, args.K, args.tag = 2_000, 4, "prueba"
    cfg = Config(a_max=args.a_max, w_step=args.w_step, w_min=args.w_min, w_max=args.w_max,
                 jac_chunk=args.jac_chunk)
    over = {} if args.u_w is None else dict(u_w=args.u_w)
    th = theta(cfg, **over)
    J, A, W, n_act, n = dims(cfg)
    outdir = os.path.join(args.outdir, args.tag)
    os.makedirs(outdir, exist_ok=True)
    print(f"dispositivo: {jax.devices()[0]} | W = {W} puntos de w, n = {n} estados por tipo, "
          f"{cfg.T * n + n_act} incógnitas por régimen", flush=True)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(args), config=dataclasses.asdict(cfg),
                       theta={k: np.asarray(v).tolist() for k, v in th.items()}), fh, indent=1)

    zs, info = solve_regimes(th, cfg, verbose=args.verbose)
    np.savez_compressed(os.path.join(outdir, "equilibrios.npz"), zs=np.stack([np.asarray(z) for z in zs]))
    objs = [equilibrium_objects(z, regime_theta(th, cfg, t), cfg) for t, z in enumerate(zs)]

    gill_age, gill_st = gillingham(cfg, outdir)
    res = [dict(regimen=t, zeta=cfg.zetas[t], **info[t], **market_stats(o, cfg)) for t, o in enumerate(objs)]
    res.append(dict(regimen="gillingham", **gill_st))
    pd.DataFrame(res).to_csv(os.path.join(outdir, "resumen.csv"), index=False)

    ages = pd.concat([pd.DataFrame(dict(regimen=t, **by_age(o, cfg))) for t, o in enumerate(objs)])
    ages = ages.merge(gill_age.rename(columns=dict(q="q_gill", P="P_gill", keep="keep_gill")),
                      on=["j", "a"], how="left")
    ages.to_csv(os.path.join(outdir, "por_edad.csv"), index=False)

    t0 = time.time()
    df = simulate_panel(objs, cfg, args.N, args.K, seed=0)
    pr = panel_resumen(df, cfg)
    pr.to_csv(os.path.join(outdir, "panel_resumen.csv"), index=False)
    if args.guardar_panel:
        df.to_csv(os.path.join(outdir, "panel.csv.gz"), index=False)

    print("\nResumen por régimen (y Gillingham):")
    print(pd.DataFrame(res).drop(columns=["ok", "iters"], errors="ignore").round(4).to_string(index=False))
    print(f"\nPor edad, régimen central contra Gillingham (marca 0 = {cfg.brands[0]}, 1 = {cfg.brands[1]}):")
    tabla(ages, gill_age, len(cfg.zetas) // 2)
    print(f"\nPanel simulado ({time.time() - t0:.1f} s):")
    print(pr.round(4).to_string(index=False))
    print(f"\nlisto: {outdir}")


if __name__ == "__main__":
    main()
