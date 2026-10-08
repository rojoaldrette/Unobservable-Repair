# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/analisis/reporte_rep0.py
# Goal:           Gráficas y tablas del reporte de la réplica 0 (docs/reporte_rep0.md)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
    python -u reporte_rep0.py

Lee claude/niu/output/modelo_tesis/montecarlo/base/ (réplica 0) y teoria/R_LB/ y escribe en
claude/niu/output/modelo_tesis/reportes/rep0/:
    fig1_sesgo_gillingham.png    sesgo relativo de los parámetros comunes: diseño 2 vs Gillingham
    fig2_reparacion.png          estimaciones ± 1.96 se de los parámetros de reparación, diseños 1-3
    fig3_teoria.png              modelo teórico: Pr(reparar) por edad y régimen; precio medio vs Gillingham
    tabla_parametros.csv         verdad, estimación y se de los 5 estimadores
'''

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "output", "modelo_tesis"))
MC = os.path.join(OUT, "montecarlo", "base")
TEO = os.path.join(OUT, "teoria", "R_LB")
FIG = os.path.join(OUT, "reportes", "rep0")

# Paleta de referencia (dataviz): 3 primeros slots categóricos, rampa azul ordinal, tintas
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
ORD = ["#86b6ef", "#2a78d6", "#104281"]
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})

NOMBRES = {"mu": "μ", "u0": "u0", "u1": "u1", "tc_buy": "Tb", "tc_buy_nocar": "Tb sin coche",
           "tc_sell": "Ts", "tc_sell_inspect": "Ts inspección", "sigma_sell": "σ vender/chatarrear",
           "acc_int": "accidente: intercepto", "acc_age": "accidente: edad", "u_w": "u_w",
           "delta": "δ", "kappa": "κ", "sigma_eta": "σ_η", "sigma_rep": "σ_rep"}
TIPOS = {"t0": "pareja", "t1": "soltero"}
MARCAS = {"j0": "LB", "j1": "HB"}


def etiqueta(p):
    for k in sorted(NOMBRES, key=len, reverse=True):
        if p == k or p.startswith(k + "_"):
            rest = p[len(k):].strip("_").split("_") if p != k else []
            extra = [TIPOS.get(r, MARCAS.get(r, r)) for r in rest if r]
            return NOMBRES[k] + (f" ({', '.join(extra)})" if extra else "")
    return p


def cargar():
    p = pd.concat([pd.read_csv(os.path.join(MC, "parametros_reps0-0.csv")),
                   pd.read_csv(os.path.join(MC, "gill_parametros_reps0-0.csv"))])
    p = p[p["mejor"]].copy()
    p["sesgo_rel"] = 100 * (p.estimado - p.verdad) / p.verdad.abs()
    p["z"] = (p.estimado - p.verdad) / p.se
    return p


def fig1(p):
    # Sesgo relativo (%) de los parámetros comunes: diseño 2 vs Gillingham parcial y completa
    est = [("diseno_2", "modelo de la tesis, diseño 2 (r no observada)", C1, "o"),
           ("gill_parcial", "Gillingham, verosimilitud parcial (la del paper)", C2, "s"),
           ("gill_completa", "Gillingham, verosimilitud completa", C3, "D")]
    comunes = p[p.estimador == "gill_parcial"].parametro.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    y = np.arange(len(comunes))[::-1]
    for k, (e, lab, c, m) in enumerate(est):
        d = p[p.estimador == e].set_index("parametro").loc[comunes]
        ax.scatter(d.sesgo_rel, y + (1 - k) * 0.22, s=30, color=c, marker=m, label=lab,
                   edgecolors=SURF, linewidths=1.0, zorder=3)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y, [etiqueta(c) for c in comunes])
    ax.set_xlabel("sesgo relativo de la estimación, % del valor verdadero")
    fig.suptitle("Ignorar la reparación sesga a Gillingham; el modelo de la tesis no se sesga",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.955), fontsize=8)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(FIG, "fig1_sesgo_gillingham.png"), dpi=170)
    plt.close(fig)


def fig2(p):
    # Parámetros de reparación: estimación ± 1.96 se por diseño, contra la verdad
    pars = ["kappa", "sigma_eta", "sigma_rep", "u_w", "delta_j0", "delta_j1"]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.6))
    for ax, par in zip(axes.ravel(), pars):
        d = p[(p.parametro == par) & p.estimador.str.startswith("diseno")].sort_values("estimador")
        x = np.arange(len(d))
        ax.errorbar(x, d.estimado, yerr=1.96 * d.se, fmt="o", color=C1, ms=6, lw=1.6, capsize=0,
                    mec=SURF, mew=1.0, zorder=3)
        ax.axhline(d.verdad.iloc[0], color=INK2, linestyle="--", linewidth=1)
        ax.set_xticks(x, ["1: ve r", "2: sin r", "3: sin r\nni motivo"])
        ax.set_title(etiqueta(par), loc="left")
        ax.margins(x=0.25)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Los tres diseños recuperan la reparación; no ver r cuesta precisión, no sesgo",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.text(0.01, 0.005, "Punto: estimación de la réplica 0.  Barra: ± 1.96 errores estándar.  "
             "Línea punteada: valor verdadero.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(FIG, "fig2_reparacion.png"), dpi=170)
    plt.close(fig)


def fig3():
    # Modelo teórico (teoria/R_LB): Pr(reparar) por edad y régimen (LB) y precio medio vs Gillingham
    d = pd.read_csv(os.path.join(TEO, "por_edad.csv"))
    lb = d[d.j == 0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for t, (z, c) in enumerate(zip(["−0.3 (barato)", "0", "+0.3 (caro)"], ORD)):
        s = lb[lb.regimen == t].query("a <= 23")
        a1.plot(s.a, s.reparar, color=c, lw=2, label=f"ζ = {z}")
        a1.annotate(f"ζ = {z}", (s.a.iloc[6], s.reparar.iloc[6]), xytext=(4, 4),
                    textcoords="offset points", color=INK2, fontsize=8)
    a1.set_xlabel("edad del coche")
    a1.set_ylabel("Pr(reparar)")
    a1.set_title("Reparación por edad, light brown", loc="left")
    a1.set_ylim(0, None)
    c = lb[lb.regimen == 1]
    a2.plot(c.a, c.P_stock, color=C1, lw=2, label="modelo de la tesis (promedio sobre w)")
    a2.plot(c.a, c.P_gill, color=C2, lw=2, label="Gillingham")
    a2.set_xlabel("edad del coche")
    a2.set_ylabel("precio de usado, miles de DKK")
    a2.set_title("Precio medio, light brown, ζ = 0", loc="left")
    a2.legend(fontsize=8)
    a2.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_teoria.png"), dpi=170)
    plt.close(fig)


def tabla(p):
    t = p.pivot_table(index="parametro", columns="estimador", values=["estimado", "se"], sort=False)
    t.insert(0, "verdad", p.groupby("parametro", sort=False).verdad.first())
    t.to_csv(os.path.join(FIG, "tabla_parametros.csv"))


def main():
    os.makedirs(FIG, exist_ok=True)
    p = cargar()
    fig1(p)
    fig2(p)
    fig3()
    tabla(p)
    print(f"listo: {FIG}")


if __name__ == "__main__":
    main()
