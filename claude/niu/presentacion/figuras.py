# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/presentacion/figuras.py
# Goal:           Gráficas de la presentación del seminario de tesis
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           27/10/2026
#
# _____________________________________________________________________________
'''
Solo lee CSV ya calculados (MC base) y fórmulas cerradas; no resuelve equilibrios.

    python figuras.py      -> figs/*.pdf

- fig_R: precio de reparación R(j, a) por régimen.
- fig_mezcla: densidad de w' sin/con reparar y la mezcla observada por régimen (Hu & Xin).
- fig_reparar: Pr(reparar) por edad en los tres regímenes (equilibrio verdadero).
- fig_sesgo: sesgo medio (%) de parámetros clave, modelo de la tesis vs Gillingham.
- fig_precios: precio por edad, verdad vs equilibrio en los θ̂ (diseño 3, Gillingham).
- fig_poblacion: distribución por edad, mismas curvas.
'''

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(AQUI, "..", "output", "modelo_tesis")
MC = os.path.join(BASE, "montecarlo", "base")
TAB = os.path.join(BASE, "reportes", "mc_base", "tabla_mc.csv")
FIGS = os.path.join(AQUI, "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "legend.fontsize": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "0.9", "figure.dpi": 150, "savefig.bbox": "tight",
})

MARCAS = ["Light brown (LB)", "Heavy brown (HB)"]
ZETAS = [-0.3, 0.0, 0.3]
C_REG = ["#1b7837", "0.15", "#c2185b"]          # R barato, base, caro
C_VERDAD, C_TESIS, C_GP, C_GC = "black", "#1f5fbf", "#e4572e", "#17a589"

# Calibración (params.py de modelo_tesis)
KAPPA, SIGMA_ETA, P_BAR = 0.25, 0.15, 0.3
R_BASE = [(4.0, 0.10, 6.0, 4.0), (10.0, 0.60, 0.0, 1.0)]


def guardar(fig, nombre):
    fig.savefig(os.path.join(FIGS, nombre + ".pdf"))
    plt.close(fig)


def etiqueta_reg(z):
    return {-0.3: r"$\zeta=-0.3$ (reparar barato)", 0.0: r"$\zeta=0$ (base)",
            0.3: r"$\zeta=+0.3$ (reparar caro)"}[z]


def fig_R():
    a = np.arange(1, 24)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    for j, ax in enumerate(axs):
        b, m, pr, tau = R_BASE[j]
        R = b + m * (a - 1) + pr * np.exp(-(a - 1) / tau)
        for z, c in zip(ZETAS, C_REG):
            ax.plot(a, R * np.exp(z), color=c, lw=2, label=etiqueta_reg(z))
        ax.set_title(MARCAS[j])
        ax.set_xlabel("edad del coche $a$ (años)")
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 32)
    axs[0].set_ylabel("$R_t(j,a)$, miles de DKK")
    axs[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Precio de reparar por marca, edad y régimen: $R_t(j,a)=R(j,a)\\,e^{\\zeta_t}$",
                 y=1.02)
    guardar(fig, "fig_R")


def fig_mezcla():
    # w = 0, light brown: m_r = w + (δ - acc_age) - κ r = κ p̄ - κ r
    x = np.linspace(-0.75, 0.6, 400)
    f0 = norm.pdf(x, KAPPA * P_BAR, SIGMA_ETA)
    f1 = norm.pdf(x, KAPPA * P_BAR - KAPPA, SIGMA_ETA)
    # tasa de reparación por régimen: promedio de Pr(r | j, a) ponderado por q (equilibrio verdadero)
    d = pd.read_csv(os.path.join(MC, "por_edad_mc.csv"))
    v = d[(d.rep == -1) & (d.estimador == "verdad") & (d.a < 24)]
    p = [np.average(v[v.regimen == t].reparar, weights=v[v.regimen == t].q) for t in range(3)]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    ax = axs[0]
    ax.plot(x, f0, color="0.3", lw=2, label="$F_0$: no reparó ($r=0$)")
    ax.plot(x, f1, color=C_TESIS, lw=2, label="$F_1$: reparó ($r=1$)")
    ax.axvline(0, color="0.6", lw=0.8, ls=":")
    ax.annotate("", xy=(KAPPA * P_BAR - KAPPA, 2.85), xytext=(KAPPA * P_BAR, 2.85),
                arrowprops=dict(arrowstyle="<->", color="0.2"))
    ax.text(KAPPA * P_BAR - KAPPA / 2, 3.0, r"$\kappa=0.25$", ha="center")
    ax.set_title("No observado: $w'$ según $r$")
    ax.set_xlabel("desgaste el año siguiente $w'$ (dado $w=0$)")
    ax.set_ylabel("densidad")
    ax.set_ylim(0, 4.3)
    ax.legend(loc="upper left", frameon=False)
    ax = axs[1]
    for z, pt, c in zip(ZETAS, p, C_REG):
        ax.plot(x, (1 - pt) * f0 + pt * f1, color=c, lw=2,
                label=etiqueta_reg(z) + f", $p_t={pt:.2f}$")
    ax.axvline(0, color="0.6", lw=0.8, ls=":")
    ax.set_title("Observado: la mezcla en cada régimen")
    ax.set_xlabel("desgaste el año siguiente $w'$ (dado $w=0$)")
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    guardar(fig, "fig_mezcla")


def fig_reparar():
    d = pd.read_csv(os.path.join(MC, "por_edad_mc.csv"))
    v = d[(d.rep == -1) & (d.estimador == "verdad")]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    for j, ax in enumerate(axs):
        for t, (z, c) in enumerate(zip(ZETAS, C_REG)):
            s = v[(v.j == j) & (v.regimen == t)].sort_values("a")
            ax.plot(s.a, s.reparar, color=c, lw=2, label=etiqueta_reg(z))
        ax.set_title(MARCAS[j])
        ax.set_xlabel("edad del coche $a$ (años)")
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 0.65)
    axs[0].set_ylabel(r"$\Pr(r=1\mid j,a)$, promedio sobre $w$")
    axs[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Probabilidad de reparar por edad (equilibrio, parámetros verdaderos)", y=1.02)
    guardar(fig, "fig_reparar")


def fig_sesgo():
    t = pd.read_csv(TAB)
    params = [("mu_t0", r"$\mu$ (pareja)"), ("mu_t1", r"$\mu$ (soltero)"),
              ("u0_t0_j0", r"$u_0$ (pareja, LB)"), ("u0_t0_j1", r"$u_0$ (pareja, HB)"),
              ("u1_t0_j0", r"$u_1$ (pareja, LB)"), ("u1_t0_j1", r"$u_1$ (pareja, HB)"),
              ("tc_buy_t0", r"$T_b$ (pareja)"), ("tc_buy_nocar_t0", r"$T_b$ sin coche (pareja)"),
              ("acc_age_j0", r"accidente: edad (LB)"),
              ("kappa", r"$\kappa$ (efecto de reparar)"), ("sigma_eta", r"$\sigma_\eta$"),
              ("delta_j0", r"$\delta$ (LB)"), ("sigma_rep", r"$\sigma_{rep}$"), ("u_w", r"$u_w$")]
    est = [("diseno_3", "Tesis, diseño 3 (sin $r$ ni motivo)", C_TESIS, "o"),
           ("gill_parcial", "Gillingham (verosimilitud del paper)", C_GP, "s"),
           ("gill_completa", "Gillingham (verosimilitud completa)", C_GC, "D")]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    y = np.arange(len(params))[::-1]
    fuera = []
    for k, (e, lab, c, mk) in enumerate(est):
        s = t[t.estimador == e].set_index("parametro")
        xs = [s.loc[p, "sesgo_pct"] if p in s.index else np.nan for p, _ in params]
        ax.scatter(np.clip(xs, -130, 130), y + 0.22 * (1 - k), color=c, marker=mk, s=34,
                   label=lab, zorder=3)
        fuera += [f"{xi:+.0f}%" for xi in xs if np.isfinite(xi) and abs(xi) > 130]
    if fuera:
        ax.text(-120, y[8] + 0.35, "Gillingham: " + " y ".join(fuera) + " (fuera de escala)",
                fontsize=11.5, color="0.3", va="bottom")
    ax.axvline(0, color="0.2", lw=1)
    ax.axhline(4.5, color="0.6", lw=0.8, ls="--")
    ax.text(129, 4.3, "solo existen en\nel modelo de la tesis", ha="right", va="top",
            fontsize=11.5, color="0.4")
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l in params])
    ax.set_xlim(-135, 135)
    ax.set_xlabel("sesgo medio en 50 réplicas, % del valor verdadero")
    ax.set_title("Sesgo medio de cada estimador (mismos paneles simulados)")
    ax.legend(loc="lower left", frameon=True, fontsize=12)
    guardar(fig, "fig_sesgo")


def perfiles(col, ylab, titulo, nombre, ylim=None, leg="upper right"):
    d = pd.read_csv(os.path.join(MC, "por_edad_mc.csv"))
    g = pd.read_csv(os.path.join(MC, "gill_por_edad_mc.csv"))
    d = d[d.regimen == 1]                      # régimen central ζ = 0
    gcol = "P" if col == "P_stock" else col
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.1))
    for j, ax in enumerate(axs):
        v = d[(d.estimador == "verdad") & (d.j == j)].sort_values("a")
        ax.plot(v.a, v[col], color=C_VERDAD, lw=2.2, label="verdad")
        series = [(d[(d.estimador == "diseno_3") & (d.j == j)], col,
                   "Tesis, diseño 3", C_TESIS),
                  (g[(g.estimador == "gill_parcial") & (g.j == j)], gcol,
                   "Gillingham (paper)", C_GP),
                  (g[(g.estimador == "gill_completa") & (g.j == j)], gcol,
                   "Gillingham (completa)", C_GC)]
        for s, c_, lab, c in series:
            q = s.groupby("a")[c_].quantile([0.05, 0.5, 0.95]).unstack()
            ax.fill_between(q.index, q[0.05], q[0.95], color=c, alpha=0.15, lw=0)
            ax.plot(q.index, s.groupby("a")[c_].mean(), color=c, lw=1.8, ls="--", label=lab)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(MARCAS[j])
        ax.set_xlabel("edad del coche $a$ (años)")
        ax.set_xlim(0, 25)
        if ylim:
            ax.set_ylim(*ylim[j])
    axs[0].set_ylabel(ylab)
    axs[1].legend(loc=leg, frameon=False)
    fig.suptitle(titulo, y=1.02)
    guardar(fig, nombre)


def fig_desgaste():
    # s = logit^-1(α0 + α1 a + w), light brown (Tabla 4 de Gillingham)
    a = np.arange(1, 25)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for w, c, lab in [(0.5, C_GP, "$w=+0.5$ (descuidado)"), (0.0, "black", "$w=0$ (curva de Gillingham)"),
                      (-0.5, C_TESIS, "$w=-0.5$ (bien cuidado)")]:
        ax.plot(a, 100 / (1 + np.exp(-(-5.625 + 0.1804 * a + w))), color=c, lw=2.2, label=lab)
    ax.set_xlabel("edad del coche $a$ (años)")
    ax.set_ylabel("prob. de avería en el año (%)")
    ax.set_title("Light brown: misma edad, distinto desgaste")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 36)
    ax.legend(loc="upper left", frameon=False)
    guardar(fig, "fig_desgaste")


def fig_calib():
    # equilibrio con parámetros verdaderos: modelo de la tesis (ζ = 0) vs Gillingham sin reparación
    d = pd.read_csv(os.path.join(MC, "por_edad_mc.csv"))
    g = pd.read_csv(os.path.join(MC, "gill_por_edad_mc.csv"))
    d = d[(d.rep == -1) & (d.estimador == "verdad") & (d.regimen == 1)]
    g = g[g.estimador == "gill_verdad"]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.9))
    for j, ax in enumerate(axs):
        s = g[g.j == j].sort_values("a")
        ax.plot(s.a, s.P, color=C_GP, lw=2.2, ls="--", label="Gillingham (sin reparación)")
        s = d[d.j == j].sort_values("a")
        ax.plot(s.a, s.P_stock, color=C_TESIS, lw=2.2, label="Tesis (con reparación)")
        ax.set_title(MARCAS[j])
        ax.set_xlabel("edad del coche $a$ (años)")
        ax.set_xlim(0, 25)
        ax.set_ylim(0, None)
    axs[0].set_ylabel("precio del usado, miles de DKK")
    axs[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Precio de equilibrio por edad con los parámetros verdaderos", y=1.02)
    guardar(fig, "fig_calib")


if __name__ == "__main__":
    fig_desgaste()
    fig_calib()
    fig_R()
    fig_mezcla()
    fig_reparar()
    fig_sesgo()
    perfiles("P_stock", "precio del usado, miles de DKK",
             "Precio por edad: verdad vs. parámetros estimados",
             "fig_precios")
    perfiles("q", "fracción de hogares $q(j,a)$",
             "Distribución por edad: verdad vs. parámetros estimados",
             "fig_poblacion", leg="lower left")
    print("listo:", sorted(os.listdir(FIGS)))
