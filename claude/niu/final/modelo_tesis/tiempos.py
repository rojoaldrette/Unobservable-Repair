# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/modelo_tesis/tiempos.py
# Goal:           Medir el costo de cada pieza de la estimación en la GPU y extrapolar al MC
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
Uso (desde claude/niu/modelo_tesis/):

    python -u tiempos.py                 # tamaño completo (GPU); usa los equilibrios de teoria/R_LB si existen
    python -u tiempos.py --prueba        # chico (CPU), para revisar que corre

Una evaluación de la verosimilitud en la estimación = para cada régimen, el equilibrio en el
θ nuevo (pasos de cuerda con la LU del θ anterior) + dz/dθ (gradiente implícito) + scores
por celda.  Este script mide cada pieza por separado:

    1. F(z) (una evaluación del sistema), armar el jacobiano, invertirlo, un producto inversa @ F
    2. dz/dθ con 4 pasos de refinamiento (los 3 regímenes)
    3. cuerda: equilibrio en θ + paso, para pasos típicos de L-BFGS, con la LU de θ, partiendo
       de z(θ) (sin predictor) o de z(θ) + dz/dθ · paso (con predictor)
    4. scores + BHHH por diseño, con un panel del tamaño del MC (N = 10,000 por régimen, K = 4)

y extrapola: horas del MC (50 réplicas × 3 diseños; Gillingham aparte, segundos) con 2 GPUs,
para 150, 250 y 400 evaluaciones por estimación.  Guarda tiempos.json en
claude/niu/output/modelo_tesis/tiempos/<tag>/.
'''

import argparse
import json
import os
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import jax
import jax.numpy as jnp

from params import Config, theta, regime_theta
from utils import dims, jacobian
from equilibrio import residual, factor, solve_chord, solve_regimes, equilibrium_objects
from gen_dataset import simulate_panel
from ll_estim import free_spec, pack, unpack, to_cells, cell_data, dz_dx, score_parts, DESIGNS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "output", "modelo_tesis"))


def timed(f, *a, reps=3):
    # (resultado, segundos promedio) sin contar la compilación (primera llamada aparte)
    out = jax.block_until_ready(f(*a))
    t0 = time.perf_counter()
    for _ in range(reps):
        out = jax.block_until_ready(f(*a))
    return out, (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prueba", action="store_true")
    ap.add_argument("--desde", default="R_LB", help="tag de teoria/ con equilibrios.npz")
    ap.add_argument("--N", type=int, default=10_000)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--tag", default="gpu")
    args = ap.parse_args()
    cfg = Config(a_max=8, w_step=0.15, w_min=-1.5, w_max=1.5, jac_chunk=128) if args.prueba else Config()
    if args.prueba:
        args.N, args.tag = 2_000, "prueba"
    th = theta(cfg)
    J, A, W, n_act, n = dims(cfg)
    nz = cfg.T * n + n_act
    R = len(cfg.zetas)
    res = dict(dispositivo=str(jax.devices()[0]), incognitas=nz, W=W)
    print(f"dispositivo: {res['dispositivo']} | {nz} incógnitas por régimen", flush=True)

    # Equilibrios verdaderos (de teoria.py si están; si no, se resuelven)
    path = os.path.join(OUT, "teoria", args.desde, "equilibrios.npz")
    if not args.prueba and os.path.exists(path) and np.load(path)["zs"].shape == (R, nz):
        zs = [jnp.asarray(z) for z in np.load(path)["zs"]]
        print(f"equilibrios de {path}", flush=True)
    else:
        zs, _ = solve_regimes(th, cfg)
    ths = [regime_theta(th, cfg, t) for t in range(R)]

    # 1. Piezas del solver
    f_res = jax.jit(lambda z, th_: residual(z, th_, cfg))
    f_jac = jax.jit(lambda z, th_: jacobian(lambda v: residual(v, th_, cfg), z, cfg.jac_chunk)[1])
    f_inv = jax.jit(jnp.linalg.inv)
    f_mv = jax.jit(lambda M, v: M @ v)
    F, res["F"] = timed(f_res, zs[1], ths[1])
    Jz, res["jacobiano"] = timed(f_jac, zs[1], ths[1], reps=1)
    Ji, res["inversa"] = timed(f_inv, Jz, reps=1)
    _, res["matvec"] = timed(f_mv, Ji, F)
    del Jz, Ji
    lus = [factor(z, th_, cfg)[1] for z, th_ in zip(zs, ths)]          # inversas por régimen
    print(f"1. F(z) {res['F'] * 1e3:.1f} ms | jacobiano {res['jacobiano']:.2f} s | "
          f"inversa {res['inversa']:.2f} s | inversa @ F {res['matvec'] * 1e3:.1f} ms", flush=True)

    # 2. dz/dθ (3 regímenes)
    spec = free_spec(th)
    x0 = pack(th, spec, cfg)
    Dz, res["dz_dx"] = timed(lambda: dz_dx(zs, lus, x0, th, cfg, spec), reps=2)
    print(f"2. dz/dθ ({len(x0)} parámetros, 3 regímenes): {res['dz_dx']:.2f} s", flush=True)

    # 3. Cuerda: equilibrio en θ + paso, con la LU de θ, sin y con predictor z + Dz Δx
    rng = np.random.default_rng(0)
    res["cuerda"] = {}
    for t in range(R):                                   # calentamiento: compilar sin medir
        solve_chord(ths[t], cfg, zs[t], lus[t])
    for step in (1e-3, 1e-2, 3e-2):
        dx = step * jnp.asarray(rng.standard_normal(len(x0)))
        th1 = unpack(x0 + dx, spec, th, cfg)
        for pred in (False, True):
            t0, stats = time.perf_counter(), []
            for t in range(R):
                z0 = zs[t] + Dz[t] @ dx if pred else zs[t]
                z1, _, ok, st = solve_chord(regime_theta(th1, cfg, t), cfg, z0, lus[t])
                jax.block_until_ready(z1)
                stats.append(dict(ok=ok, **st))
            dt = time.perf_counter() - t0
            key = f"{step:g}" + ("_predictor" if pred else "")
            res["cuerda"][key] = dict(segundos=dt, regimenes=stats)
            print(f"3. cuerda, paso {step:g}, {'con' if pred else 'sin'} predictor: {dt:.2f} s | "
                  + " | ".join(f"{s['cuerda']} pasos, {s['factorizaciones']} refact."
                               + (", NEWTON COMPLETO" if s["newton_completo"] else "")
                               + ("" if s["ok"] else ", NO CONVERGE") for s in stats), flush=True)

    # 4. Scores + BHHH por diseño, panel del tamaño del MC
    objs = [equilibrium_objects(z, th_, cfg) for z, th_ in zip(zs, ths)]
    df = simulate_panel(objs, cfg, args.N, args.K, seed=0)
    res["scores"], res["celdas"] = {}, {}
    for design in DESIGNS:
        cells = to_cells(df, cfg, design)
        data = cell_data(cells, cfg)
        _, dt = timed(lambda: score_parts(zs, Dz, x0, th, data, cfg, spec, design), reps=2)
        res["scores"][design], res["celdas"][design] = dt, len(cells)
        print(f"4. scores + BHHH, diseño {design}: {dt:.2f} s ({len(cells):,} celdas, "
              f"{int(cells['cnt'].sum()):,} obs.)", flush=True)

    try:
        res["memoria_pico_GB"] = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        res["memoria_pico_GB"] = None

    # Extrapolación: una evaluación = cuerda con predictor (paso típico 1e-2) + dz/dθ + scores
    per_eval = {d: res["cuerda"]["0.01_predictor"]["segundos"] + res["dz_dx"] + res["scores"][d] for d in DESIGNS}
    res["por_evaluacion"] = per_eval
    print(f"\nPor evaluación (cuerda con paso 1e-2 + dz/dθ + scores): "
          + ", ".join(f"diseño {d}: {v:.2f} s" for d, v in per_eval.items()), flush=True)
    print("MC completo (50 réplicas × 3 diseños, 2 GPUs con un proceso cada una):")
    res["horas_mc"] = {}
    for n_eval in (150, 250, 400):
        h = 50 * sum(per_eval.values()) * n_eval / 2 / 3600
        res["horas_mc"][n_eval] = h
        print(f"   {n_eval} evaluaciones por estimación: {h:.1f} h")
    if res["memoria_pico_GB"]:
        print(f"memoria pico en la GPU: {res['memoria_pico_GB']:.1f} GB")

    outdir = os.path.join(OUT, "tiempos", args.tag)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "tiempos.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(f"listo: {outdir}")


if __name__ == "__main__":
    main()
