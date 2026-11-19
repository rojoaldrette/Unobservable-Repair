# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/modelo_tesis/por_edad_mc.py
# Goal:           Equilibrio en cada θ estimado del MC, por (marca, edad): precios,
#                 Pr(reparar) y distribución de la población (para reporte_mc.py)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
Uso (desde claude/niu/modelo_tesis/; tamaño completo en GPU):

    python -u por_edad_mc.py --reps 0:50 > por_edad.log 2>&1
    python -u por_edad_mc.py --smoke              # juguete (CPU), sobre montecarlo/smoke

El MC guardó los θ̂ pero no los equilibrios.  Aquí, para cada (réplica, diseño), se arma θ̂
con los valores naturales de parametros_reps*.csv y se resuelve el equilibrio de los 3
regímenes con pasos de cuerda desde el verdadero (predictor z + dz/dx Δx, inversa del
jacobiano verdadero).  Gillingham (gill_parcial, gill_completa) se resuelve en
niu/gillingham/por_edad_mc.py, en un proceso aparte en CPU, al mismo tiempo.

Salidas en claude/niu/output/modelo_tesis/montecarlo/<tag>/ (se agregan filas; lo que ya
está se salta, así que se puede cortar y volver a correr):
    por_edad_mc.csv        rep, estimador, regimen, j, a, q, P_stock, P_transaccion, keep,
                           reparar, w_medio  (equilibrium.by_age; rep = -1, estimador = verdad
                           para el equilibrio verdadero)
    gill_por_edad_mc.csv   rep, estimador, j, a, q, P, keep  (rep = -1, estimador = gill_verdad:
                           Gillingham en los parámetros verdaderos)
q es la masa de hogares que empieza el año con un coche (j, a), sumada sobre w y tipos.
'''

import argparse
import glob
import os
import subprocess
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from params import Config, theta, regime_theta
from equilibrio import factor, solve_chord, equilibrium_objects, by_age
from ll_estim import free_spec, pack, labels, dz_dx
from montecarlo import true_equilibria, OUT, GILL


def cargar_parametros(mc, patron):
    # Mejor arranque; si un (rep, estimador) se re-estimó en *_faltantes, vale ese (como reporte_mc.py)
    files = sorted(glob.glob(os.path.join(mc, patron)), key=lambda f: ("_faltantes" in f, f))
    p = pd.concat([pd.read_csv(f).assign(archivo=f) for f in files], ignore_index=True)
    p = p[p["mejor"]]
    return p[p.archivo == p.groupby(["rep", "estimador"]).archivo.transform("last")]


def theta_hat(est, th0, spec):
    # θ̂ a partir de los valores naturales (en el orden de labels(spec))
    th, i = dict(th0), 0
    for k, shape in spec:
        size = int(np.prod(shape))
        th[k] = jnp.asarray(est[i:i + size]).reshape(shape)
        i += size
    return th


def filas(rep, estimador, zs, th, cfg):
    out = []
    for t, z in enumerate(zs):
        d = by_age(equilibrium_objects(z, regime_theta(th, cfg, t), cfg), cfg)
        out.append(pd.DataFrame(dict(rep=rep, estimador=estimador, regimen=t, **d)))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0:50")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--desde", default="R_LB", help="tag de teoria/ con los equilibrios verdaderos")
    ap.add_argument("--sin_gillingham", dest="gillingham", action="store_false")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        cfg = Config(a_max=8, w_step=0.15, w_min=-1.5, w_max=1.5, jac_chunk=128)
        args.tag, args.desde = "smoke", "-"
    else:
        cfg = Config()
    mc = os.path.join(OUT, "montecarlo", args.tag)
    a, b = map(int, args.reps.split(":"))
    path = os.path.join(mc, "por_edad_mc.csv")
    hechos = set()
    if os.path.exists(path):
        h = pd.read_csv(path, usecols=["rep", "estimador"])
        hechos = set(zip(h.rep, h.estimador))

    gill = None
    if args.gillingham:
        cmd = [sys.executable, "-u", "por_edad_mc.py", "--mc", mc, "--reps", args.reps,
               "--a_max", str(cfg.a_max), "--brands", ",".join(cfg.brands),
               "--types", ",".join(cfg.types), "--nocar", cfg.nocar]
        gill = subprocess.Popen(cmd, cwd=GILL, env=dict(os.environ, JAX_PLATFORMS="cpu"))

    th0 = theta(cfg)
    spec = free_spec(th0)
    labs = labels(spec)
    x0 = pack(th0, spec, cfg)
    t0 = time.time()
    zs_true = true_equilibria(th0, cfg, args.desde, False)
    Jinvs = [factor(z, regime_theta(th0, cfg, t), cfg)[1] for t, z in enumerate(zs_true)]
    Dz = dz_dx(zs_true, Jinvs, x0, th0, cfg, spec)
    print(f"dispositivo: {jax.devices()[0]} | verdad lista ({time.time() - t0:.0f} s)", flush=True)
    if (-1, "verdad") not in hechos:
        filas(-1, "verdad", zs_true, th0, cfg).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    p = cargar_parametros(mc, "parametros_reps*.csv")
    p = p[(p.rep >= a) & (p.rep < b)]
    for (rep, est), d in p.groupby(["rep", "estimador"]):
        if (rep, est) in hechos:
            continue
        t1 = time.time()
        th = theta_hat(d.set_index("parametro").loc[labs, "estimado"].to_numpy(), th0, spec)
        x = pack(th, spec, cfg)
        zs, ok_all = [], True
        for t in range(len(cfg.zetas)):
            z, _, ok, _ = solve_chord(regime_theta(th, cfg, t), cfg, zs_true[t] + Dz[t] @ (x - x0), Jinvs[t])
            zs.append(z)
            ok_all &= bool(ok)
        if not ok_all:
            print(f"  OJO: rep {rep} [{est}] el equilibrio no converge; se omite", flush=True)
            continue
        filas(int(rep), est, zs, th, cfg).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"rep {rep} [{est}] listo ({time.time() - t1:.1f} s)", flush=True)

    if gill is not None and gill.wait() != 0:
        print("OJO: falló la parte de Gillingham", flush=True)
    print(f"listo: {mc} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
