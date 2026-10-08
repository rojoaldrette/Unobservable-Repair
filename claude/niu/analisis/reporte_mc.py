# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/analisis/reporte_mc.py
# Goal:           Gráficas y tablas del Monte Carlo completo (docs/reporte_mc.md)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
    python -u reporte_mc.py [--tag base]

Solo lee CSV (corre en local en segundos).  Lee claude/niu/output/modelo_tesis/montecarlo/<tag>/
(todos los parametros_reps*.csv y resumen_reps*.csv, incluidos los *_faltantes.csv de
montecarlo.py --solo_faltantes; si un (rep, estimador) aparece dos veces se queda el último
archivo) y escribe en claude/niu/output/modelo_tesis/reportes/mc_<tag>/:
    fig1_sesgo.png           sesgo relativo medio de los 21 parámetros comunes: Gillingham vs diseños
                             (izquierda) y zoom a los diseños con IC 95% del sesgo medio (derecha)
    fig2_reparacion.png      distribución entre réplicas de los parámetros de reparación, por diseño
    fig3_cobertura.png       cobertura del IC 95% por parámetro y estimador, con banda binomial
    fig4_se.png              sd entre réplicas vs se mediano (calibración de los errores estándar)
    fig5_fallas.png          qué (réplica, diseño) no tienen estimación
    tabla_mc.csv / .md / .tex  por estimador y parámetro: verdad, media, sesgo, sesgo %, t del
                             sesgo, sd, se mediano, sd/se, RMSE, cobertura, réplicas
    tabla_estimadores.csv / .md  por estimador: réplicas, fallas, evaluaciones, segundos, etc.
'''

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reporte_rep0 import etiqueta, C1, C2, C3, ORD, INK, INK2, MUTED, GRID, BASE, SURF  # noqa: F401 (estilo)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "output", "modelo_tesis"))

# Diseños: rampa azul (más oscuro = menos información); Gillingham: naranja y aqua
EST = {"diseno_1": ("diseño 1: ve r y motivo", ORD[0], "o"),
       "diseno_2": ("diseño 2: sin r", ORD[1], "o"),
       "diseno_3": ("diseño 3: sin r ni motivo", ORD[2], "o"),
       "gill_parcial": ("Gillingham parcial (la del paper)", C2, "s"),
       "gill_completa": ("Gillingham completa", C3, "D")}
DISENOS = ["diseno_1", "diseno_2", "diseno_3"]
REPARACION = ["kappa", "sigma_eta", "sigma_rep", "u_w", "delta_j0", "delta_j1"]


# Datos ______________________________________________________________

def leer(mc, patron):
    files = sorted(glob.glob(os.path.join(mc, patron)), key=lambda f: ("_faltantes" in f, f))
    return pd.concat([pd.read_csv(f).assign(archivo=os.path.basename(f)) for f in files], ignore_index=True)


def cargar(mc):
    p = pd.concat([leer(mc, "parametros_reps*.csv"), leer(mc, "gill_parametros_reps*.csv")], ignore_index=True)
    p = p[p["mejor"]]
    # un (rep, estimador) re-estimado en *_faltantes reemplaza al anterior
    ultimo = p.groupby(["rep", "estimador"]).archivo.transform("last")
    p = p[p.archivo == ultimo].copy()
    p["err"] = p.estimado - p.verdad
    p["cubre"] = (p.estimado - 1.96 * p.se <= p.verdad) & (p.verdad <= p.estimado + 1.96 * p.se)
    r = leer(mc, "resumen_reps*.csv").drop_duplicates(["rep", "estimador"], keep="last")
    g = leer(mc, "gill_resumen_reps*.csv").drop_duplicates(["rep", "estimador"], keep="last")
    return p, r, g


def tabla(p):
    g = p.groupby(["estimador", "parametro"], sort=False)
    R = g.rep.nunique()
    t = pd.DataFrame(dict(verdad=g.verdad.first(), media=g.estimado.mean(), sesgo=g.err.mean(),
                          sd=g.estimado.std(), se_mediano=g.se.median(),
                          rmse=g.err.apply(lambda e: np.sqrt(np.mean(e ** 2))),
                          cobertura=g.cubre.mean(), reps=R))
    t.insert(3, "sesgo_pct", 100 * t.sesgo / t.verdad.abs())
    t.insert(4, "t_sesgo", t.sesgo / (t.sd / np.sqrt(t.reps)))
    t.insert(7, "sd_se", t.sd / t.se_mediano)
    return t.reset_index()


def tabla_estimadores(p, r, g, reps_total):
    rows = []
    for e in EST:
        d = r[r.estimador == e] if e.startswith("diseno") else g[g.estimador == e]
        n = d.rep.nunique()
        rows.append(dict(estimador=e, estimaciones=n, perdidas=reps_total - n,
                         convergio=d.convergio.mean(), evals=d.n_eval.mean(), segundos=d.segundos.mean(),
                         fallas_dentro=d.n_fail.mean() if "n_fail" in d else np.nan,
                         P_rmse=d.P_rmse.mean() if "P_rmse" in d else np.nan,
                         reparacion_verdad=d.reparacion_verdad.mean() if "reparacion_verdad" in d else np.nan,
                         reparacion_est=d.reparacion_est.mean() if "reparacion_est" in d else np.nan))
    return pd.DataFrame(rows)


# Gráficas ______________________________________________________________

def fig1(t, fig_dir):
    comunes = t[t.estimador == "gill_parcial"].parametro.tolist()
    y = np.arange(len(comunes))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 6.6), sharey=True,
                                 gridspec_kw=dict(width_ratios=[1.25, 1]))
    off = dict(zip(EST, [0.28, 0.14, 0.0, -0.14, -0.28]))
    for e, (lab, c, m) in EST.items():
        d = t[t.estimador == e].set_index("parametro").loc[comunes]
        a1.scatter(d.sesgo_pct, y + off[e], s=26, color=c, marker=m, label=lab,
                   edgecolors=SURF, linewidths=0.8, zorder=3)
        if e in DISENOS:
            ic = 100 * 1.96 * d.sd / np.sqrt(d.reps) / d.verdad.abs()
            a2.errorbar(d.sesgo_pct, y + off[e], xerr=ic, fmt=m, color=c, ms=5, lw=1.4,
                        capsize=0, mec=SURF, mew=0.8, zorder=3)
    for a in (a1, a2):
        a.axvline(0, color=INK2, linewidth=1)
        a.tick_params(axis="y", length=0)
    a1.set_yticks(y, [etiqueta(c) for c in comunes])
    a1.set_xlabel("sesgo medio, % del valor verdadero")
    a1.set_title("Los cinco estimadores", loc="left")
    a2.set_xlabel("sesgo medio, % (barra: IC 95% del sesgo medio)")
    a2.set_title("Zoom: solo el modelo de la tesis", loc="left")
    fig.suptitle("Ignorar la reparación sesga a Gillingham; los tres diseños del modelo de la tesis no se sesgan",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(fig_dir, "fig1_sesgo.png"), dpi=170)
    plt.close(fig)


def fig2(p, fig_dir):
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.0))
    rng = np.random.default_rng(0)
    for ax, par in zip(axes.ravel(), REPARACION):
        for k, e in enumerate(DISENOS):
            d = p[(p.parametro == par) & (p.estimador == e)]
            if d.empty:
                continue
            c = EST[e][1]
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(d)), d.estimado, s=10, color=c, alpha=0.55,
                       edgecolors="none", zorder=2)
            ax.plot([k - 0.3, k + 0.3], [d.estimado.mean()] * 2, color=INK, lw=2, zorder=3)
        tv = p[p.parametro == par].verdad.iloc[0]
        ax.axhline(tv, color=INK2, linestyle="--", linewidth=1, zorder=1)
        ax.set_xticks(range(3), ["1: ve r", "2: sin r", "3: sin r\nni motivo"])
        ax.set_title(etiqueta(par), loc="left")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Parámetros de reparación: centrados en la verdad; sin r la dispersión crece poco",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.text(0.01, 0.005, "Punto: una réplica.  Raya negra: media entre réplicas.  "
             "Línea punteada: valor verdadero.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(fig_dir, "fig2_reparacion.png"), dpi=170)
    plt.close(fig)


def fig3(t, fig_dir):
    pars = t[t.estimador == "diseno_1"].parametro.tolist()
    y = np.arange(len(pars))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 7.6))
    R = int(t[t.estimador.isin(DISENOS)].reps.min())
    lo, hi = 0.95 - 1.96 * np.sqrt(0.95 * 0.05 / R), min(1.0, 0.95 + 1.96 * np.sqrt(0.95 * 0.05 / R))
    ax.axvspan(lo, hi, color=GRID, alpha=0.7, zorder=0, lw=0)
    ax.axvline(0.95, color=INK2, linewidth=1)
    off = dict(zip(EST, [0.28, 0.14, 0.0, -0.14, -0.28]))
    for e, (lab, c, m) in EST.items():
        d = t[t.estimador == e].set_index("parametro").reindex(pars)
        ax.scatter(d.cobertura, y + off[e], s=24, color=c, marker=m, label=lab,
                   edgecolors=SURF, linewidths=0.8, zorder=3)
    ax.set_yticks(y, [etiqueta(c) for c in pars])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("cobertura del intervalo ± 1.96 se (fracción de réplicas que contienen la verdad)")
    fig.suptitle("Cobertura ~95% en los diseños; Gillingham casi nunca cubre la verdad",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.text(0.01, 0.005, f"Banda gris: 0.95 ± 1.96 errores binomiales con {R} réplicas.  "
             "Los parámetros de reparación solo existen en los diseños.", color=MUTED, fontsize=8)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    fig.savefig(os.path.join(fig_dir, "fig3_cobertura.png"), dpi=170)
    plt.close(fig)


def fig4(t, fig_dir):
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    d = t[t.estimador.isin(DISENOS)]
    for e in DISENOS:
        s = d[d.estimador == e]
        ax.scatter(s.se_mediano, s.sd, s=22, color=EST[e][1], label=EST[e][0],
                   edgecolors=SURF, linewidths=0.8, zorder=3)
    v = np.array([d[["se_mediano", "sd"]].min().min() * 0.7, d[["se_mediano", "sd"]].max().max() * 1.4])
    ax.plot(v, v, color=INK2, lw=1, zorder=1)
    ax.fill_between(v, 0.8 * v, 1.25 * v, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("error estándar mediano (BHHH)")
    ax.set_ylabel("desviación estándar entre réplicas")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Los errores estándar miden bien la dispersión real", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.text(0.01, 0.005, "Un punto por parámetro.  Línea: sd = se.  Banda: sd/se entre 0.8 y 1.25.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(fig_dir, "fig4_se.png"), dpi=170)
    plt.close(fig)


def fig5(r, reps, fig_dir):
    fig, ax = plt.subplots(figsize=(8.6, 1.9))
    hechos = set(zip(r.rep, r.estimador))
    for k, e in enumerate(DISENOS):
        ok = [rep for rep in reps if (rep, e) in hechos]
        mal = [rep for rep in reps if (rep, e) not in hechos]
        ax.scatter(ok, [k] * len(ok), marker="s", s=34, color=BASE, edgecolors="none", zorder=2)
        ax.scatter(mal, [k] * len(mal), marker="s", s=34, color=C2, edgecolors="none", zorder=3)
        ax.text(reps[-1] + 1.2, k, f"{len(mal)} perdidas", va="center", color=INK2, fontsize=8)
    ax.set_yticks(range(3), ["diseño 1", "diseño 2", "diseño 3"])
    ax.set_ylim(2.6, -0.6)
    ax.set_xlim(reps[0] - 1, reps[-1] + 7)
    ax.set_xlabel("réplica")
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle("Estimaciones perdidas (naranja): el equilibrio no convergió al arrancar BHHH",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(fig_dir, "fig5_fallas.png"), dpi=170)
    plt.close(fig)


# Tablas en texto ______________________________________________________________

def a_md(df, fmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(c, row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def formato(c, v):
    if isinstance(v, str):
        return etiqueta(v) if c == "parametro" else v
    if c in ("reps", "estimaciones", "perdidas"):
        return f"{int(v)}"
    if c in ("sesgo_pct",):
        return f"{v:+.1f}"
    if c in ("t_sesgo", "sd_se", "cobertura", "convergio"):
        return f"{v:.2f}"
    if c in ("evals", "segundos"):
        return f"{v:.0f}"
    return f"{v:.4g}"


def escribir_tablas(t, te, fig_dir):
    t.to_csv(os.path.join(fig_dir, "tabla_mc.csv"), index=False)
    te.to_csv(os.path.join(fig_dir, "tabla_estimadores.csv"), index=False)
    cols = ["parametro", "verdad", "sesgo_pct", "t_sesgo", "sd_se", "cobertura", "reps"]
    with open(os.path.join(fig_dir, "tabla_mc.md"), "w", encoding="utf-8") as fh:
        for e, (lab, _, _) in EST.items():
            fh.write(f"### {lab}\n\n" + a_md(t[t.estimador == e][cols], formato) + "\n")
    with open(os.path.join(fig_dir, "tabla_estimadores.md"), "w", encoding="utf-8") as fh:
        fh.write(a_md(te, formato))
    # LaTeX: sesgo % y cobertura, una columna por estimador
    w = t.pivot_table(index="parametro", columns="estimador", values=["sesgo_pct", "cobertura"], sort=False)
    orden = t[t.estimador == "diseno_1"].parametro.tolist()
    with open(os.path.join(fig_dir, "tabla_mc.tex"), "w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{l" + "rr" * len(EST) + "}\n\\hline\n")
        fh.write(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{EST[e][0]}}}" for e in EST) + " \\\\\n")
        fh.write("parámetro & " + " & ".join(["sesgo \\% & cob."] * len(EST)) + " \\\\\n\\hline\n")
        for par in orden:
            cells = []
            for e in EST:
                s, c = w.get(("sesgo_pct", e), {}).get(par, np.nan), w.get(("cobertura", e), {}).get(par, np.nan)
                cells += ["--", "--"] if np.isnan(s) else [f"{s:+.1f}", f"{c:.2f}"]
            fh.write(etiqueta(par).replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()
    mc = os.path.join(OUT, "montecarlo", args.tag)
    fig_dir = os.path.join(OUT, "reportes", f"mc_{args.tag}")
    os.makedirs(fig_dir, exist_ok=True)
    p, r, g = cargar(mc)
    reps = sorted(set(g.rep) | set(r.rep))
    t = tabla(p)
    te = tabla_estimadores(p, r, g, len(reps))
    fig1(t, fig_dir)
    fig2(p, fig_dir)
    fig3(t, fig_dir)
    fig4(t, fig_dir)
    fig5(r, reps, fig_dir)
    escribir_tablas(t, te, fig_dir)
    print(te.round(3).to_string(index=False))
    print(f"listo: {fig_dir}")


if __name__ == "__main__":
    main()
