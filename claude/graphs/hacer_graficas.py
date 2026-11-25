# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/graphs/hacer_graficas.py
# Goal:           Tablas y gráficas de resultados del Monte Carlo para la tesis
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           22/11/2026
#
# _____________________________________________________________________________
'''
Uso (desde la raíz del repo o desde cualquier lado):

    python -u claude/graphs/hacer_graficas.py            # todo + compila graficas.tex
    python -u claude/graphs/hacer_graficas.py --sin_pdf  # sin compilar el .tex

Lee (no escribe nada fuera de claude/graphs/):
    claude/niu/final/output/modelo_tesis/montecarlo/<tag>/
        parametros_reps*.csv, gill_parametros_reps*.csv   estimados por réplica
        por_edad_mc.csv, gill_por_edad_mc.csv             equilibrio en cada θ̂, por (marca, edad)
    claude/niu/final/output/modelo_tesis/teoria/R_LB/
        equilibrios.npz                                   z verdadero de los 3 regímenes

Escribe en claude/graphs/:
    tablas/tab_comunes.tex, tablas/tab_propios.tex       tablas 1 y 2
    tablas/resumen_parametros.csv                        todas las estadísticas por parámetro
    tablas/cifras.tex                                    \\newcommand con los números del texto
    figuras/*.pdf                                        figuras 3 a 8
    graficas.pdf                                         (compila graficas.tex)

Las gráficas 7 y 8 evalúan el modelo (CCPs, transición de w y distribución estacionaria)
en el equilibrio verdadero ya resuelto: no resuelven ningún equilibrio (unos segundos en CPU).
'''

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
FINAL = RAIZ / "claude/niu/final"
sys.path.insert(0, str(FINAL / "modelo_tesis"))

FIG = AQUI / "figuras"
TAB = AQUI / "tablas"

# ---------------------------------------------------------------- estilo
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.7,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.grid": True,
    "grid.color": "#E3E3E3",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})
ANCHO = 6.0  # pulgadas; en el .tex se incluyen a \textwidth

# Estimadores: orden, nombre, color, marcador, línea (color + forma: nunca solo color)
ESTS = ["gill_completa", "diseno_1", "diseno_2", "diseno_3"]
NOMBRE = {"verdad": "Verdadero", "gill_completa": "Gillingham", "diseno_1": "Diseño 1",
          "diseno_2": "Diseño 2", "diseno_3": "Diseño 3"}
# Paleta de la tesis ["#002F2A","#1E5B4F","#d6af6b","#a5712c","#9B2247","#611232"] y tonos:
# verdes = diseños, dorado = Gillingham, guinda = verdad.
COLOR = {"verdad": "#611232", "gill_completa": "#C29548", "diseno_1": "#002F2A",
         "diseno_2": "#1E5B4F", "diseno_3": "#5E9C8A"}
MARCA = {"gill_completa": "s", "diseno_1": "o", "diseno_2": "^", "diseno_3": "D"}
LINEA = {"gill_completa": "-", "diseno_1": "-", "diseno_2": (0, (5, 2)),
         "diseno_3": (0, (1.5, 1.2))}
BANDA_VERDAD = "#EBD3DB"
REG = ["Reparación barata", "Reparación base", "Reparación cara"]
REG_CORTO = ["Reparación barata", "Régimen base", "Reparación cara"]
REG_MINI = ["Barata", "Base", "Cara"]
REG_COLOR = ["#1E5B4F", "#a5712c", "#9B2247"]
REG_LINEA = ["-", (0, (5, 2)), (0, (1.5, 1.5))]
# Figura 6: una familia de color por régimen; dentro de ella un tono por diseño (oscuro -> claro)
REG_TONOS = [["#002F2A", "#1E5B4F", "#6FA595"],     # barata: verdes
             ["#6B4718", "#a5712c", "#d6af6b"],     # base: dorados
             ["#611232", "#9B2247", "#CF7F98"]]     # cara: guindas
REG_BANDA = ["#D2E4DE", "#F3E6CC", "#F1D8E0"]
GRIS_DISENO = {"diseno_1": "#333333", "diseno_2": "#777777", "diseno_3": "#AAAAAA"}
MARCAS_COCHE = ["Light brown", "Heavy brown"]

# Parámetros: etiqueta LaTeX y descripción (notación de tesis.tex, tabla tab:parametros)
J_ = {"j0": r"\text{LB}", "j1": r"\text{HB}"}
T_ = {"t0": "1", "t1": "2"}


def etiqueta(p):
    s = p.split("_")
    if p.startswith("mu_"):
        return rf"$\mu_{{{T_[s[1]]}}}$", "Utilidad marginal del dinero"
    if p.startswith("u0_"):
        return rf"$u_{{0,{T_[s[1]]}{J_[s[2]]}}}$", "Utilidad base"
    if p.startswith("u1_"):
        return rf"$u_{{1,{T_[s[1]]}{J_[s[2]]}}}$", "Utilidad: edad"
    if p.startswith("tc_buy_nocar"):
        return rf"$T^{{\emptyset}}_{{b,{T_[s[3]]}}}$", "Costo del comprador sin coche"
    if p.startswith("tc_buy"):
        return rf"$T_{{b,{T_[s[2]]}}}$", "Costo del comprador"
    if p == "tc_sell":
        return r"$T_s$", "Costo del vendedor"
    if p == "sigma_sell":
        return r"$\sigma_D$", "Escala de vender o chatarrizar"
    if p.startswith("acc_int"):
        return rf"$\alpha_{{0,{J_[s[2]]}}}$", "Accidentes: constante"
    if p.startswith("acc_age"):
        return rf"$\alpha_{{1,{J_[s[2]]}}}$", "Accidentes: edad"
    if p == "u_w":
        return r"$u_w$", "Utilidad del desgaste"
    if p.startswith("delta"):
        return rf"$\delta_{{{J_[s[1]]}}}$", "Desgaste anual sin reparar"
    if p == "kappa":
        return r"$\kappa$", "Efecto de reparar"
    if p == "sigma_eta":
        return r"$\sigma_\eta$", "D.E.\\ del choque al desgaste"
    if p == "sigma_rep":
        return r"$\sigma_r$", "Escala de la decisión de reparar"
    raise KeyError(p)


COMUNES = ["mu_t0", "mu_t1", "u0_t0_j0", "u0_t0_j1", "u0_t1_j0", "u0_t1_j1",
           "u1_t0_j0", "u1_t0_j1", "u1_t1_j0", "u1_t1_j1", "tc_buy_t0", "tc_buy_t1",
           "tc_buy_nocar_t0", "tc_buy_nocar_t1", "tc_sell", "sigma_sell",
           "acc_int_j0", "acc_int_j1", "acc_age_j0", "acc_age_j1"]
PROPIOS = ["u_w", "delta_j0", "delta_j1", "kappa", "sigma_eta", "sigma_rep"]
GRUPO = {**{p: "Preferencias" for p in COMUNES[:10]},
         **{p: "Costos de transacción" for p in COMUNES[10:16]},
         **{p: "Accidentes" for p in COMUNES[16:]},
         "u_w": "Preferencias", "delta_j0": "Desgaste", "delta_j1": "Desgaste",
         "kappa": "Desgaste", "sigma_eta": "Desgaste", "sigma_rep": "Reparación"}


# ---------------------------------------------------------------- datos del MC
def cargar_parametros(mc, patron):
    """Una fila por (rep, estimador, parametro): el mejor arranque; si una réplica se cortó y
    se volvió a estimar (en el mismo archivo o en uno de reanudación, o en *_faltantes), vale
    la última.  Igual que cargar_parametros de por_edad_mc.py + quitar duplicados."""
    files = sorted(glob.glob(str(mc / patron)), key=lambda f: ("_faltantes" in f, f))
    p = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    p["mejor"] = p["mejor"].astype(str) == "True"
    p["convergio"] = p["convergio"].astype(str) == "True"
    p = p[p["mejor"]]
    return p.drop_duplicates(["rep", "estimador", "parametro"], keep="last")


def estadisticas(p):
    p = p.assign(dif=p.estimado - p.verdad,
                 cubre=(p.estimado - p.verdad).abs() <= 1.96 * p.se)
    g = p.groupby(["estimador", "parametro"])
    s = pd.DataFrame(dict(
        n=g.size(), verdad=g.verdad.first(), media=g.estimado.mean(),
        de=g.estimado.std(ddof=1), sesgo=g.dif.mean(),
        rmse=g.dif.apply(lambda d: np.sqrt((d ** 2).mean())),
        se_mediano=g.se.median(), cobertura=g.cubre.mean()))
    s["sesgo_pct"] = 100 * s.sesgo / s.verdad.abs()
    s["de_pct"] = 100 * s.de / s.verdad.abs()
    return s.reset_index()


def cargar_por_edad(mc):
    pe = pd.read_csv(mc / "por_edad_mc.csv").drop_duplicates(
        ["rep", "estimador", "regimen", "j", "a"], keep="last")
    ge = pd.read_csv(mc / "gill_por_edad_mc.csv").drop_duplicates(
        ["rep", "estimador", "j", "a"], keep="last")
    ge = ge[ge.estimador == "gill_completa"]
    return pe, ge


# ---------------------------------------------------------------- tablas 1 y 2
def num(x, d=3):
    x = round(x, d) + 0.0          # sin "-0.0"
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if x < 0 else s


def estrellas(media, de):
    """Estrellas de z = media/DE contra cero (dos colas): * 10%, ** 5%, *** 1%."""
    z = float(abs(media) / de) if de > 0 else np.inf
    k = sum(int(z >= c) for c in (1.645, 1.960, 2.576))
    return rf"$^{{{'*' * int(k)}}}$" if k else ""


def tabla(stats, params, ests, archivo, titulo, etiq, nota):
    S = stats.set_index(["estimador", "parametro"])
    ncol = 2 + 2 * len(ests)
    cab1 = " & ".join([r"\multicolumn{2}{c}{" + NOMBRE[e] + "}" for e in ests])
    cmid = " ".join([rf"\cmidrule(lr){{{3 + 2 * k}-{4 + 2 * k}}}" for k in range(len(ests))])
    cab2 = " & ".join([r"Media & Sesgo \%"] * len(ests))
    filas, grupo = [], None
    for prm in params:
        if GRUPO[prm] != grupo:
            grupo = GRUPO[prm]
            if filas:
                filas.append(r"\addlinespace")
            filas.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{grupo}}}}}\\")
        lab, _ = etiqueta(prm)
        v = S.loc[(ests[0], prm), "verdad"]
        celdas, celdas_de = [lab, num(v, 3)], ["", ""]
        for e in ests:
            r = S.loc[(e, prm)]
            sp = r.sesgo_pct
            sp_txt = num(sp, 1)
            if abs(sp) >= 5:
                sp_txt = r"\textbf{" + sp_txt + "}"
            celdas += [num(r.media, 3) + estrellas(r.media, r.de), sp_txt]
            celdas_de += ["(" + num(r.de, 3) + ")", ""]
        filas.append(" & ".join(celdas) + r"\\")
        filas.append(" & ".join(celdas_de) + r"\\[1pt]")
    reps = ", ".join(f"{NOMBRE[e]}: {int(S.loc[(e, params[0]), 'n'])}" for e in ests)
    tex = rf"""\begin{{table}}[htbp]
\centering
\small
\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{l r {' '.join(['rr'] * len(ests))}}}
\toprule
 & & {cab1}\\
{cmid}
Parámetro & Verdad & {cab2}\\
\midrule
{chr(10).join(filas)}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\caption{{{titulo}}}
\label{{{etiq}}}
\par\smallskip
\begin{{minipage}}{{\textwidth}}\footnotesize
\textit{{Nota:}} {nota} Media de $\hat\theta$ entre réplicas;
entre paréntesis, su desviación estándar entre réplicas (en unidades del parámetro).
Significancia de $\overline{{\hat\theta}}/\text{{DE}}$ contra cero (normal, dos colas):
$^{{*}}p<0.10$, $^{{**}}p<0.05$, $^{{***}}p<0.01$.
Sesgo \% $=100\,(\overline{{\hat\theta}}-\theta_0)/|\theta_0|$; en negritas, $|\text{{sesgo}}|\ge5\%$.
Réplicas: {reps}.
\end{{minipage}}
\end{{table}}
"""
    (TAB / archivo).write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------- figura 3
def fig_dispersion(stats):
    S = stats.set_index(["estimador", "parametro"])
    # una figura por grupo de parámetros (como las tablas 1 y 2); eje en ±LIM %
    figuras = [("fig3a_dispersion_comunes.pdf", COMUNES, ESTS, 50.0),
               ("fig3b_dispersion_propios.pdf", PROPIOS, ESTS[1:], 10.0)]
    for archivo, params, ests, LIM in figuras:
        H = 0.24 * len(params) + 1.0 + (0.3 if len(params) < 10 else 0)
        fig, ax = plt.subplots(figsize=(ANCHO, H))
        fig.subplots_adjust(top=1 - 0.45 / H, bottom=0.55 / H)
        n = len(ests)
        off = np.linspace(-0.27, 0.27, n) if n > 1 else [0]
        for i, prm in enumerate(params):
            y = len(params) - 1 - i
            if i % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#F5F6F8", zorder=0, lw=0)
            for k, e in enumerate(ests):
                r = S.loc[(e, prm)]
                m, d = r.sesgo_pct, r.de_pct
                yy = y + off[k]
                lo, hi = max(m - d, -LIM), min(m + d, LIM)
                if hi > -LIM and lo < LIM:
                    ax.plot([lo, hi], [yy, yy], color=COLOR[e], lw=1.6, solid_capstyle="round",
                            zorder=3)
                if abs(m) <= LIM:
                    ax.plot(m, yy, MARCA[e], color=COLOR[e], ms=4.5, mec="white", mew=0.6,
                            zorder=4)
                else:   # fuera de escala: flecha en el borde con el valor
                    xb = np.sign(m) * LIM
                    ax.plot(xb, yy, ">" if m > 0 else "<", color=COLOR[e], ms=5, zorder=4,
                            clip_on=False)
                    ax.text(xb - np.sign(m) * 0.05 * LIM, yy + 0.22, f"${m:.0f}\\%$", color="#333333",
                            fontsize=7.5, va="bottom", ha="left" if m < 0 else "right")
        ax.axvline(0, color=COLOR["verdad"], lw=1.0, zorder=2)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels([etiqueta(p)[0] for p in params[::-1]])
        ax.set_ylim(-0.6, len(params) - 0.4)
        ax.set_xlim(-LIM, LIM)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(r"Desviación respecto al valor verdadero, $100\,(\hat\theta-\theta_0)/|\theta_0|$ (\%)")
        h = [Line2D([], [], color=COLOR[e], marker=MARCA[e], lw=1.6, ms=5, mec="white",
                    label=NOMBRE[e]) for e in ests]
        fig.legend(handles=h, loc="upper center", ncol=len(ests), bbox_to_anchor=(0.55, 1.0))
        fig.savefig(FIG / archivo)
        plt.close(fig)


# ---------------------------------------------------------------- figuras 4 a 6
def banda(df, col, por):
    g = df.groupby(por)[col]
    return g.mean(), g.quantile(0.025), g.quantile(0.975)


def fig_por_edad(pe, ge, col, gcol, archivo, ylabel, regimenes=(1,), ymin=None, amax=None,
                 ancho_banda=("diseno_3", "gill_completa")):
    filas = len(regimenes)
    fig, axs = plt.subplots(filas, 2, figsize=(ANCHO, (2.55 if filas == 1 else 2.15) * filas + 0.5), sharex=True,
                            squeeze=False)
    for i, t in enumerate(regimenes):
        for j in range(2):
            ax = axs[i, j]
            d = pe[(pe.regimen == t) & (pe.j == j)]
            if amax:
                d = d[d.a <= amax]
            v = d[d.estimador == "verdad"].sort_values("a")
            ax.plot(v.a, v[col], color=BANDA_VERDAD, lw=6, solid_capstyle="round", zorder=1)
            ax.plot(v.a, v[col], color=COLOR["verdad"], lw=0.9, zorder=2)
            series = [(e, d[d.estimador == e], col) for e in ESTS[1:]]
            if gcol is not None:
                g = ge[ge.j == j]
                if amax:
                    g = g[g.a <= amax]
                series = [("gill_completa", g, gcol)] + series
            for e, de, c in series:
                m, lo, hi = banda(de, c, "a")
                if e in ancho_banda:
                    ax.fill_between(m.index, lo, hi, color=COLOR[e], alpha=0.16, lw=0, zorder=2)
                ax.plot(m.index, m, color=COLOR[e], ls=LINEA[e], lw=1.3, marker=MARCA[e],
                        ms=3, markevery=3, mec="white", mew=0.4, zorder=3)
            if ymin is not None:
                ax.set_ylim(bottom=ymin)
            tit = MARCAS_COCHE[j] + (f", {REG_CORTO[t].lower()}" if filas > 1 else "")
            ax.set_title(tit, loc="left", fontsize=9.5)
            if j == 0:
                ax.set_ylabel(ylabel)
        axs[-1, 0].set_xlabel("Edad del coche, $a$")
        axs[-1, 1].set_xlabel("Edad del coche, $a$")
    h = [Line2D([], [], color=BANDA_VERDAD, lw=6, label=NOMBRE["verdad"])]
    if gcol is not None:
        h.append(Line2D([], [], color=COLOR["gill_completa"], marker="s", ms=4,
                        label=NOMBRE["gill_completa"]))
    h += [Line2D([], [], color=COLOR[e], ls=LINEA[e], marker=MARCA[e], ms=4, label=NOMBRE[e])
          for e in ESTS[1:]]
    fig.legend(handles=h, loc="upper center", ncol=len(h),
               bbox_to_anchor=(0.5, 1.0 + 0.22 / fig.get_figheight()))
    fig.tight_layout()
    fig.savefig(FIG / archivo)
    plt.close(fig)


def fig_reparar_regimenes(pe, archivo="fig6_reparar_edad.pdf", amax=23):
    """Figura 6: Pr(reparar | j, a), los tres regímenes en el mismo eje (un panel por marca).
    Color = régimen (familia verde / dorada / guinda); tono y trazo = diseño; banda = verdad."""
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO, 3.0), sharex=True)
    for j, ax in enumerate(axs):
        for t in range(3):
            d = pe[(pe.regimen == t) & (pe.j == j) & (pe.a <= amax)]
            v = d[d.estimador == "verdad"].sort_values("a")
            ax.plot(v.a, v.reparar, color=REG_BANDA[t], lw=6, solid_capstyle="round", zorder=1)
            for k, e in enumerate(ESTS[1:]):
                m = d[d.estimador == e].groupby("a").reparar.mean()
                ax.plot(m.index, m, color=REG_TONOS[t][k], ls=LINEA[e], lw=1.2,
                        marker=MARCA[e], ms=3, markevery=(k, 3), mec="white", mew=0.4,
                        zorder=3)
        ax.set_ylim(bottom=0)
        ax.set_title(MARCAS_COCHE[j], loc="left", fontsize=9.5)
        ax.set_xlabel("Edad del coche, $a$")
    axs[0].set_ylabel(r"$\Pr(\text{reparar}\mid j,a)$")
    # leyenda en dos renglones: régimen (color) y diseño (tono/trazo, en grises)
    h_reg = [Line2D([], [], color=REG_TONOS[t][1], lw=5, label=REG[t]) for t in range(3)]
    h_dis = [Line2D([], [], color="#BBBBBB", lw=6, label=NOMBRE["verdad"])]
    h_dis += [Line2D([], [], color=GRIS_DISENO[e], ls=LINEA[e], marker=MARCA[e], ms=4, lw=1.2,
                     label=NOMBRE[e]) for e in ESTS[1:]]
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.legend(handles=h_reg, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.legend(handles=h_dis, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.92))
    fig.savefig(FIG / archivo)
    plt.close(fig)


# ---------------------------------------------------------------- figuras 7 y 8 (modelo)
def objetos_verdaderos():
    from params import Config, theta, regime_theta
    from equilibrio import equilibrium_objects
    from utils import split_states, dims
    cfg = Config()
    th = theta(cfg)
    zs = np.load(FINAL / "output/modelo_tesis/teoria/R_LB/equilibrios.npz")["zs"]
    J, A, W, n_act, n = dims(cfg)
    reg = []
    for t in range(len(cfg.zetas)):
        objs = equilibrium_objects(zs[t], regime_theta(th, cfg, t), cfg)
        tipos = []
        for o, f in zip(objs, cfg.f):
            q = o["q"]
            qa = split_states(q, cfg)[0]
            keep = split_states(o["keep"], cfg)[0]
            m = q @ o["trade"]
            hold = qa * keep + m * o["buy"][:n_act].reshape(J, A - 1, W)   # tenencias usadas
            tipos.append(dict(f=f, hold=hold, q=qa, rep=o["repair"], F=o["F"], s=o["s"]))
        reg.append(tipos)
    return cfg, reg


def pooled_rep(tipos):
    """Pr(r=1 | j, d, w) juntando tipos con el peso de las tenencias f_tau * h_tau(j, d, w)."""
    num = sum(t["f"] * t["hold"] * t["rep"] for t in tipos)
    den = sum(t["f"] * t["hold"] for t in tipos)
    return num, den


def fig_mezcla(cfg, reg, j=0, d=8, w=0.0):
    g = cfg.w_grid
    h = cfg.w_step
    k = int(np.argmin(np.abs(g - w)))
    F = reg[0][0]["F"]                       # F no depende del régimen ni del tipo
    f0, f1 = F[0, j, k] / h, F[1, j, k] / h  # densidades en el grid
    pis = []
    for tipos in reg:
        num, den = pooled_rep(tipos)
        pis.append(float(num[j, d - 1, k] / den[j, d - 1, k]))
    xlo, xhi = w - 0.75, w + 0.6
    sel = (g >= xlo) & (g <= xhi)
    # (a) mezclas observadas
    fig, ax = plt.subplots(figsize=(ANCHO, 2.9))
    for t, p in enumerate(pis):
        z = p * f1 + (1 - p) * f0
        ax.plot(g[sel], z[sel], color=REG_COLOR[t], ls=REG_LINEA[t], lw=1.6, label=REG[t])
    ax.axvline(w, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("Desgaste el año siguiente, $w'$")
    ax.set_ylabel("Densidad")
    ax.set_ylim(bottom=0)
    fig.tight_layout(rect=(0, 0, 1, 0.88))       # la leyenda va arriba, fuera de los ejes
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(FIG / "fig7a_mezcla_observada.pdf")
    plt.close(fig)
    # (b) componentes reales
    C_REP, C_NOREP, C_MEZ = "#1E5B4F", "#d6af6b", "#611232"
    fig, axs = plt.subplots(1, 3, figsize=(ANCHO, 2.6), sharey=True)
    for t, (ax, p) in enumerate(zip(axs, pis)):
        c1, c0 = p * f1, (1 - p) * f0
        ax.fill_between(g[sel], c1[sel], color=C_REP, alpha=0.35, lw=0)
        ax.fill_between(g[sel], c0[sel], color=C_NOREP, alpha=0.45, lw=0)
        ax.plot(g[sel], c1[sel], color=C_REP, lw=1.1)
        ax.plot(g[sel], c0[sel], color="#a5712c", lw=1.1, ls=(0, (4, 1.5)))
        ax.plot(g[sel], (c0 + c1)[sel], color=C_MEZ, lw=1.1)
        ax.set_title(REG_MINI[t], loc="left", fontsize=9.5)
        ax.set_xlabel("$w'$")
        ax.axvline(w, color="#999999", lw=0.8, ls=":")
    tope = max(float(((1 - p) * f0 + p * f1)[sel].max()) for p in pis)
    axs[0].set_ylim(0, 1.06 * tope)
    axs[0].set_ylabel("Densidad")
    hnd = [Line2D([], [], color=C_REP, lw=5, alpha=0.6, label="Reparó"),
           Line2D([], [], color=C_NOREP, lw=5, alpha=0.8, label="No reparó"),
           Line2D([], [], color=C_MEZ, lw=1.1, label="Mezcla observada")]
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(handles=hnd, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(FIG / "fig7b_componentes.pdf")
    plt.close(fig)
    return pis, g[k]


def fig_rep_w(cfg, reg, d_fijo=8):
    g = cfg.w_grid
    A = cfg.a_max
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO, 2.7), sharey=True)
    pend = {}
    for j, ax in enumerate(axs):
        for t, tipos in enumerate(reg):
            num, den = pooled_rep(tipos)
            # edad fija: así la curva es el efecto de w y no de la composición por edades
            n_ = num[j, d_fijo - 1]
            d_ = den[j, d_fijo - 1]
            p = n_ / np.maximum(d_, 1e-300)
            masa = d_ / d_.sum()
            # rango: donde está el 99% central de las tenencias de este régimen
            cdf = np.cumsum(masa)
            sel = (cdf >= 0.005) & (cdf <= 0.995)
            ax.plot(g[sel], p[sel], color=REG_COLOR[t], ls=REG_LINEA[t], lw=1.6,
                    label=REG[t])
            # efecto de un aumento de 0.1 en w (= 2/3 de sigma_eta), promedio sobre la masa
            dp = np.gradient(p, g)
            pend[(j, t)] = float((dp * masa).sum() * 0.1)
            # punto: tasa promedio y w medio del régimen
            wm = (masa * g).sum()
            pm = (masa * p).sum()
            ax.plot(wm, pm, "o", color=REG_COLOR[t], ms=5, mec="white", mew=0.8, zorder=5)
        ax.set_title(f"{MARCAS_COCHE[j]}, edad $a={d_fijo}$", loc="left", fontsize=9.5)
        ax.set_xlabel("Desgaste, $w$")
    axs[0].set_ylabel(r"$\Pr(\text{reparar}\mid w)$")
    axs[0].set_ylim(0, None)
    hnd, lab = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(hnd, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(FIG / "fig8_reparar_desgaste.pdf")
    plt.close(fig)
    return pend


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--sin_pdf", action="store_true")
    args = ap.parse_args()
    mc = FINAL / "output/modelo_tesis/montecarlo" / args.tag
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)

    # --- parámetros
    p = pd.concat([cargar_parametros(mc, "parametros_reps*.csv"),
                   cargar_parametros(mc, "gill_parametros_reps*.csv")], ignore_index=True)
    conv = p.groupby("estimador").apply(lambda d: d.groupby("rep").convergio.first().mean())
    n_tot = p.groupby("estimador").rep.nunique()
    p = p[p.convergio]
    stats = estadisticas(p)
    stats.to_csv(TAB / "resumen_parametros.csv", index=False, float_format="%.6g")
    print("réplicas (convergidas) por estimador:", p.groupby("estimador").rep.nunique().to_dict())

    tabla(stats, COMUNES, ESTS, "tab_comunes.tex",
          "Parámetros que el modelo comparte con Gillingham et al.\\ (2022): estimados del "
          "Monte Carlo", "tab:mc-comunes",
          "Gillingham se estima con su verosimilitud ``completa'' (observa el motivo de salida) "
          "sobre el mismo panel, sin $w$ ni $r$.")
    tabla(stats, PROPIOS, ESTS[1:], "tab_propios.tex",
          "Parámetros propios del modelo con reparación: estimados del Monte Carlo",
          "tab:mc-propios",
          "Gillingham no tiene estos parámetros.")
    fig_dispersion(stats)

    # --- por edad
    pe, ge = cargar_por_edad(mc)
    n_pe = pe[pe.rep >= 0].groupby("estimador").rep.nunique().to_dict()
    n_ge = ge.rep.nunique()
    print("réplicas en por_edad:", n_pe, "| gillingham:", n_ge)
    fig_por_edad(pe, ge, "P_transaccion", "P", "fig4_precios.pdf",
                 "Precio del usado (miles DKK)")
    fig_por_edad(pe, ge, "q", "q", "fig5_poblacion.pdf",
                 "Masa de hogares con el coche, $q(j,a)$", ymin=0)
    fig_reparar_regimenes(pe)

    # errores de precio y q (para el texto)
    v = pe[(pe.estimador == "verdad") & (pe.regimen == 1)].set_index(["j", "a"])
    err = {}
    for e in ESTS[1:]:
        d = pe[(pe.estimador == e) & (pe.regimen == 1)].groupby(["j", "a"]).P_transaccion.mean()
        err[e] = float((100 * (d / v.P_transaccion - 1)).abs().max())
    gm = ge.groupby(["j", "a"]).P.mean()
    err_g = (100 * (gm / v.P_transaccion - 1))

    # --- modelo en la verdad
    cfg, reg = objetos_verdaderos()
    pis, w_x = fig_mezcla(cfg, reg)
    pend = fig_rep_w(cfg, reg)
    print("Pi(r=1|x) por régimen:", [round(x, 3) for x in pis], "| pendientes:", pend)

    # --- cifras para el .tex
    S = stats.set_index(["estimador", "parametro"])
    peor = {e: S.loc[e].loc[COMUNES + PROPIOS].sesgo_pct.abs().max() for e in ESTS[1:]}
    cob = {e: S.loc[e].loc[COMUNES + PROPIOS].cobertura for e in ESTS[1:]}
    c = {
        "NrepsDuno": int(n_tot["diseno_1"]), "NrepsDdos": int(n_tot["diseno_2"]),
        "NrepsDtres": int(n_tot["diseno_3"]), "NrepsGill": int(n_tot["gill_completa"]),
        "ConvDuno": f"{100 * conv['diseno_1']:.1f}", "ConvDdos": f"{100 * conv['diseno_2']:.1f}",
        "ConvDtres": f"{100 * conv['diseno_3']:.1f}", "ConvGill": f"{100 * conv['gill_completa']:.1f}",
        "PeorSesgoD": f"{max(peor.values()):.2f}",
        "CobMin": f"{100 * min(x.min() for x in cob.values()):.1f}",
        "CobMax": f"{100 * max(x.max() for x in cob.values()):.1f}",
        "NrepsEdad": int(min(n_pe[e] for e in ESTS[1:])), "NrepsEdadGill": int(n_ge),
        "ErrPrecioD": f"{max(err.values()):.2f}",
        "ErrPrecioGillJoven": f"{err_g.loc[(0, 1)]:.1f}",
        "PiBarata": f"{pis[0]:.2f}", "PiBase": f"{pis[1]:.2f}", "PiCara": f"{pis[2]:.2f}",
        "PendLBBarata": f"{pend[(0, 0)]:.3f}", "PendLBBase": f"{pend[(0, 1)]:.3f}",
        "PendLBCara": f"{pend[(0, 2)]:.3f}", "PendHBBarata": f"{pend[(1, 0)]:.3f}",
        "PendHBBase": f"{pend[(1, 1)]:.3f}", "PendHBCara": f"{pend[(1, 2)]:.3f}",
    }
    for prm in ["mu_t0", "mu_t1", "tc_sell", "u0_t0_j0", "acc_age_j0", "tc_buy_t0"]:
        c["Gill" + prm.replace("_", "").replace("0", "cero").replace("1", "uno")] = \
            f"{S.loc[('gill_completa', prm), 'sesgo_pct']:.0f}"
    txt = "% generado por hacer_graficas.py: no editar a mano\n" + "\n".join(
        rf"\newcommand{{\{k}}}{{{str(v).replace('-', '$-$')}}}" for k, v in c.items()) + "\n"
    (TAB / "cifras.tex").write_text(txt, encoding="utf-8")
    print(txt)

    if not args.sin_pdf:
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "graficas.tex"],
                               cwd=AQUI, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
        print("pdflatex:", "ok" if r.returncode == 0 else r.stdout[-3000:])


if __name__ == "__main__":
    main()
