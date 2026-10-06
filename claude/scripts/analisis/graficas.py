# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/analisis/graficas.py
# Goal:           Gráficas de los resultados de estimación (PNG y PDF)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           06/10/2026
#
# _____________________________________________________________________________
'''
Cada función recibe corridas (datos.Corrida) y una carpeta de salida, y guarda PNG
(200 dpi) y PDF.  Colores fijos por fuente (nunca por orden): verdad en negro, D0 azul,
D1 naranja, Gillingham aqua / violeta.  Superficies y mapas de calor: una sola rampa
secuencial (azules).

    distribucion_edad      q por edad y marca (suma sobre s y tipos), todas las fuentes
    sin_coche              fracción de hogares sin coche por fuente
    distribucion_s         mapa de calor q(a, s) por marca y fuente (modelo_fin)
    precios_3d_<marca>     superficie P(a, s) por fuente (modelo_fin); celdas sin masa en blanco
    precios_edad           media de P sobre s ponderada por q, por edad y marca; con Gillingham
    ccps_edad              Pr(reparar), Pr(keep) y Pr(chatarrear) por edad (ponderadas por q)
    mc_sesgo               (si hay varias réplicas) estimado − verdad por parámetro y estimador
'''

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra la proyección 3d)

from datos import etiqueta, agregar_tipos, media_en_s, MARCAS, mejores, unir

COLOR = {
    "verdad": "#0b0b0b", "verdad_gill": "#52514e",
    "oraculo": "#2a78d6", "hx": "#eb6834",
    "gill_parcial": "#1baf7a", "gill_completa": "#4a3aa7",
    "gill_parcial|modelo_fin": "#e87ba4", "gill_completa|modelo_fin": "#eda100",
}
ESTILO = {"verdad": dict(lw=2.4, ls="-"), "verdad_gill": dict(lw=1.6, ls=":")}
MARCADOR = {"oraculo": "o", "hx": "s", "gill_parcial": "^", "gill_completa": "v",
            "gill_parcial|modelo_fin": "^", "gill_completa|modelo_fin": "v"}
SECUENCIAL = "Blues"


def _eje_s(c, df):
    # Con s en log-odds el grid es uniforme en ℓ = logit(s): se grafica contra ℓ
    logodds = c.config.get("params", {}).get("s_space") == "logodds"
    if logodds and "ell" in df:
        return "ell", "ℓ = logit(s)"
    return "s", "s (prob. de descompostura)"
ORDEN_LEYENDA = ("verdad", "oraculo", "hx", "gill_parcial|modelo_fin", "gill_completa|modelo_fin",
                 "verdad_gill", "gill_parcial", "gill_completa")

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b2a", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "legend.frameon": False,
})


def _guardar(fig, out, nombre):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"{nombre}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out, f"{nombre}.pdf"), bbox_inches="tight")
    plt.close(fig)


def _linea(ax, x, y, fuente):
    kw = dict(color=COLOR.get(fuente, "#8a8984"), label=etiqueta(fuente))
    kw.update(ESTILO.get(fuente, dict(lw=1.6, ls="--", marker=MARCADOR.get(fuente, "o"), ms=4)))
    ax.plot(x, y, **kw)


def _marcas(df):
    return sorted(int(j) for j in df["j"].unique() if j >= 0)


def _leyenda(fig, axes):
    # A la derecha de la figura, fuera de los ejes (no tapa etiquetas)
    vistos = {}
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            vistos.setdefault(l, h)
    orden = [etiqueta(f) for f in ORDEN_LEYENDA]
    labs = sorted(vistos, key=lambda l: orden.index(l) if l in orden else len(orden))
    fig.legend([vistos[l] for l in labs], labs, loc="center left", bbox_to_anchor=(1.0, 0.5))


# Distribución ______________________________________________________________

def _dist_edad(c):
    d = c.distribucion
    if "regimen" in d:
        d = d[d["regimen"] == d["regimen"].min()]   # modelo_fin guarda solo el régimen central
    d = d[d["estado"] == "activo"]
    return agregar_tipos(d, c.f, claves=("fuente", "j", "a"))


def distribucion_edad(mf, gills, out):
    tablas = ([_dist_edad(mf)] if mf is not None and mf.distribucion is not None else [])
    tablas += [_dist_edad(g) for g in gills if g.distribucion is not None]
    if not tablas:
        return
    d = unir(tablas)
    marcas = _marcas(d)
    fig, axes = plt.subplots(1, len(marcas), figsize=(4 * len(marcas), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, j in zip(axes, marcas):
        for fuente, dd in d[d["j"] == j].groupby("fuente", sort=False):
            dd = dd.sort_values("a")
            _linea(ax, dd["a"], dd["q"], fuente)
        ax.set_title(MARCAS.get(j, f"marca {j}"))
        ax.set_xlabel("edad del coche (años)")
    axes[0].set_ylabel("fracción de hogares (q)")
    _leyenda(fig, axes)
    fig.suptitle("Distribución de coches por edad", y=1.02)
    _guardar(fig, out, "distribucion_edad")


def sin_coche(mf, gills, out):
    rows = []
    for c in ([mf] if mf is not None else []) + list(gills):
        if c.distribucion is None:
            continue
        d = c.distribucion
        if "regimen" in d:
            d = d[d["regimen"] == d["regimen"].min()]
        d = d[d["estado"] == "sin_coche"]
        w = d["tipo"].map(dict(enumerate(c.f)))
        rows += [(f, float((dd["q"] * w[dd.index]).sum())) for f, dd in d.groupby("fuente", sort=False)
                 if f not in {r[0] for r in rows}]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.45 * len(rows)))
    names = [etiqueta(f) for f, _ in rows]
    vals = [v for _, v in rows]
    ax.barh(names, vals, color=[COLOR.get(f, "#8a8984") for f, _ in rows], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", color="#2b2b2a")
    ax.invert_yaxis()
    ax.set_xlabel("fracción de hogares sin coche")
    ax.grid(axis="y", visible=False)
    _guardar(fig, out, "sin_coche")


def distribucion_s(mf, out):
    if mf is None or mf.distribucion is None:
        return
    d = mf.distribucion[mf.distribucion["estado"] == "activo"]
    ycol, ylab = _eje_s(mf, d)
    d = agregar_tipos(d, mf.f, claves=("fuente", "j", "a", ycol))
    fuentes = list(dict.fromkeys(d["fuente"]))
    marcas = _marcas(d)
    vmax = d["q"].max()
    fig, axes = plt.subplots(len(fuentes), len(marcas), figsize=(3.8 * len(marcas), 2.8 * len(fuentes)),
                             squeeze=False, sharex=True, sharey=True)
    for i, fuente in enumerate(fuentes):
        for k, j in enumerate(marcas):
            dd = d[(d["fuente"] == fuente) & (d["j"] == j)].pivot(index=ycol, columns="a", values="q")
            ax = axes[i, k]
            im = ax.pcolormesh(dd.columns, dd.index, dd.values, cmap=SECUENCIAL, vmin=0, vmax=vmax,
                               shading="nearest")
            ax.grid(False)
            if i == 0:
                ax.set_title(MARCAS.get(j, f"marca {j}"))
            if k == 0:
                ax.set_ylabel(f"{etiqueta(fuente)}\n{ylab}")
            if i == len(fuentes) - 1:
                ax.set_xlabel("edad")
    fig.colorbar(im, ax=axes, shrink=0.8, label="fracción de hogares (q)")
    _guardar(fig, out, "distribucion_s")


# Precios ______________________________________________________________

def _regimen(df, regimen):
    if regimen is None:
        regimen = int(np.median(df["regimen"].unique()))
    return df[df["regimen"] == regimen], regimen


def precios_3d(mf, out, regimen=None):
    if mf is None or mf.precios is None:
        return
    d, regimen = _regimen(mf.precios, regimen)
    ycol, ylab = _eje_s(mf, d)
    fuentes = list(dict.fromkeys(d["fuente"]))
    d = d.assign(P_plot=np.where(d["sin_masa"].astype(str).str.lower() == "true", np.nan, d["P"]))
    for j in _marcas(d):
        dj = d[d["j"] == j]
        zmin, zmax = np.nanmin(dj["P_plot"]), np.nanmax(dj["P_plot"])
        # Eje s recortado a donde hay masa en alguna fuente (el resto del grid queda vacío)
        s_con_masa = dj.loc[dj["P_plot"].notna(), ycol]
        s_lo, s_hi = s_con_masa.min(), s_con_masa.max()
        dj = dj[(dj[ycol] >= s_lo) & (dj[ycol] <= s_hi)]
        fig = plt.figure(figsize=(5.2 * len(fuentes), 4.8))
        for i, fuente in enumerate(fuentes):
            p = dj[dj["fuente"] == fuente].pivot(index=ycol, columns="a", values="P_plot")
            A, Sg = np.meshgrid(p.columns.to_numpy(float), p.index.to_numpy(float))
            ax = fig.add_subplot(1, len(fuentes), i + 1, projection="3d")
            ax.plot_surface(A, Sg, p.values, cmap=SECUENCIAL, vmin=zmin, vmax=zmax,
                            edgecolor="#ffffff", linewidth=0.2, antialiased=True)
            ax.set_zlim(zmin, zmax)
            ax.set_xlabel("edad a", labelpad=6)
            ax.set_ylabel(ylab, labelpad=8)
            ax.set_zlabel("P (miles DKK)", labelpad=8)
            ax.set_title(etiqueta(fuente))
            ax.view_init(elev=25, azim=-135)
        fig.subplots_adjust(wspace=0.15, left=0.02, right=0.96)
        fig.suptitle(f"Precios de usados P(a, s): {MARCAS.get(j, j)} (régimen {regimen}; "
                     "s = prob. de descompostura; celdas sin masa en blanco)", y=1.0)
        _guardar(fig, out, f"precios_3d_marca{j}")


def precios_edad(mf, gills, out, regimen=None):
    tablas = []
    if mf is not None and mf.precios is not None:
        d, _ = _regimen(mf.precios, regimen)
        tablas.append(media_en_s(d, "P"))
    for g in gills:
        if g.precios is not None:
            tablas.append(g.precios[["fuente", "j", "a", "P"]])
    if not tablas:
        return
    d = unir(tablas)
    marcas = _marcas(d)
    fig, axes = plt.subplots(1, len(marcas), figsize=(4 * len(marcas), 3.4))
    axes = np.atleast_1d(axes)
    for ax, j in zip(axes, marcas):
        for fuente, dd in d[d["j"] == j].groupby("fuente", sort=False):
            dd = dd.sort_values("a")
            _linea(ax, dd["a"], dd["P"], fuente)
        ax.set_title(MARCAS.get(j, f"marca {j}"))
        ax.set_xlabel("edad del coche (años)")
    axes[0].set_ylabel("P (miles DKK), media sobre s ponderada por q")
    _leyenda(fig, axes)
    fig.suptitle("Precios de usados por edad", y=1.02)
    _guardar(fig, out, "precios_edad")


# CCPs ______________________________________________________________

def _ccps_edad(c, cols):
    d = c.ccps
    w = d["q"] * d["tipo"].map(dict(enumerate(c.f)))
    d = d.assign(_w=w)
    d = d[d["_w"] > 0]
    out = []
    for col in cols:
        if col not in d:
            continue
        num = d.assign(_n=d[col] * d["_w"]).groupby(["fuente", "j", "a"])["_n"].sum()
        den = d.groupby(["fuente", "j", "a"])["_w"].sum()
        out.append((num / den).rename(col))
    return pd.concat(out, axis=1).reset_index() if out else None


def ccps_edad(mf, gills, out):
    tablas = []
    if mf is not None and mf.ccps is not None:
        tablas.append(_ccps_edad(mf, ("repair", "keep", "scrap")))
    tablas += [_ccps_edad(g, ("keep", "scrap")) for g in gills if g.ccps is not None]
    tablas = [t for t in tablas if t is not None]
    if not tablas:
        return
    d = unir(tablas)
    marcas = _marcas(d)
    filas = [("repair", "Pr(reparar | coche)"), ("keep", "Pr(quedarse el coche)"),
             ("scrap", "Pr(chatarrear | se deshace)")]
    filas = [f for f in filas if f[0] in d]
    fig, axes = plt.subplots(len(filas), len(marcas), figsize=(4 * len(marcas), 3.0 * len(filas)),
                             squeeze=False, sharex=True)
    for i, (col, nombre) in enumerate(filas):
        for k, j in enumerate(marcas):
            ax = axes[i, k]
            for fuente, dd in d[(d["j"] == j) & d[col].notna()].groupby("fuente", sort=False):
                dd = dd.sort_values("a")
                _linea(ax, dd["a"], dd[col], fuente)
            if i == 0:
                ax.set_title(MARCAS.get(j, f"marca {j}"))
            if k == 0:
                ax.set_ylabel(nombre)
            if i == len(filas) - 1:
                ax.set_xlabel("edad del coche (años)")
    _leyenda(fig, axes)
    _guardar(fig, out, "ccps_edad")


# Monte Carlo ______________________________________________________________

def mc_sesgo(corridas, out, parametros=None, max_param=12):
    d = pd.concat([mejores(c.parametros) for c in corridas if c.parametros is not None])
    if d.empty or d["rep"].nunique() < 2:
        return
    if parametros is None:
        parametros = list(dict.fromkeys(d["parametro"]))[:max_param]
    d = d[d["parametro"].isin(parametros)].assign(err=lambda x: x["estimado"] - x["verdad"])
    estimadores = list(dict.fromkeys(d["estimador"]))
    n = len(parametros)
    ncol = min(4, n)
    fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(3.4 * ncol, 2.8 * np.ceil(n / ncol)),
                             squeeze=False)
    for ax, par in zip(axes.ravel(), parametros):
        datos = [d[(d["parametro"] == par) & (d["estimador"] == e)]["err"].dropna() for e in estimadores]
        bp = ax.boxplot(datos, patch_artist=True, widths=0.6, showfliers=False)
        for patch, e in zip(bp["boxes"], estimadores):
            patch.set_facecolor(COLOR.get(e, "#8a8984"))
            patch.set_alpha(0.75)
        ax.axhline(0, color="#0b0b0b", lw=1)
        ax.set_xticks(range(1, len(estimadores) + 1))
        ax.set_xticklabels([etiqueta(e) for e in estimadores], rotation=30, ha="right", fontsize=8)
        ax.set_title(par, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Monte Carlo: estimado − verdad", y=1.01)
    _guardar(fig, out, "mc_sesgo")
