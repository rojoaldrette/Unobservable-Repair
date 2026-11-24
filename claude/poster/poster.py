# -*- coding: utf-8 -*-
"""Póster de tesis (90 x 120 cm vertical: 3 columnas + banda inferior).

Uso:     python claude/poster/poster.py  [--mc DIR] [--precios DIR] [--sin-png]
Salida:  claude/poster/poster_tesis.pptx  y  claude/poster/poster_tesis.png

- Las ecuaciones, los dos árboles y los párrafos con matemáticas se escriben en LaTeX
  (cadenas TEX_* más abajo) y se compilan con pdflatex + pdftocairo a PNG transparente
  (caché en _tex/; solo se recompila lo que cambió).
- Las gráficas se hacen con matplotlib (caché en _fig/, se rehacen siempre).
- Al final se exporta el PNG con PowerPoint (render.ps1); --sin-png lo omite.
  Si el .pptx está abierto en PowerPoint, ciérralo antes de correr.

Datos: parámetros estimados de --mc (parametros_reps*.csv, gill_parametros_reps*.csv);
precios estimados de --precios (por_edad_mc.csv, gill_por_edad_mc.csv, que escribe
modelo_tesis/por_edad_mc.py). Si --precios no los tiene, se usa la corrida smoke.
"""
import argparse
import csv
import glob
import hashlib
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
MCDIR = RAIZ / "claude/niu/final/output/modelo_tesis/montecarlo"

ap = argparse.ArgumentParser()
ap.add_argument("--mc", default=str(MCDIR / "base"))
ap.add_argument("--precios", default=None,
                help="carpeta con por_edad_mc.csv (default: la de --mc)")
ap.add_argument("--out", default=str(AQUI / "poster_tesis.pptx"))
ap.add_argument("--sin-png", action="store_true")
ARGS = ap.parse_args()

TEXDIR = AQUI / "_tex"
FIGDIR = AQUI / "_fig"
TEXDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- estilo
HEX = dict(navy="17243A", blue="2A5CAA", orange="D2691E", cream="FCEFE3",
           pale="EEF2F8", text="1E232B", muted="5F6873", rule="C8D0DA",
           sky="9DB9E3", redact="3C4552", oscuro="A34A0E")
NAVY, BLUE, ORANGE, CREAM, PALE, TEXT, MUTED, RULE, SKY, REDACT = (
    RGBColor.from_string(HEX[k]) for k in
    ("navy", "blue", "orange", "cream", "pale", "text", "muted", "rule", "sky",
     "redact"))
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SOFT = RGBColor(0xC9, 0xD3, 0xE3)

FONT = "Palatino Linotype"      # el mismo Palatino que newpxtext en LaTeX
B, S, XS = 27, 23, 20           # cuerpo, secundario, notas (pt)

W_IN, H_IN = 90 / 2.54, 120 / 2.54
MARGIN, GAP = 1.0, 0.9
NCOL = 3
COLW = (W_IN - 2 * MARGIN - (NCOL - 1) * GAP) / NCOL
COLX = [MARGIN + k * (COLW + GAP) for k in range(NCOL)]
CM = 2.54
Y0 = 6.35


def I(x):
    return Inches(x)


# ---------------------------------------------------------------- LaTeX -> PNG
PREAMBULO = r"""\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[spanish,es-noshorthands,es-nodecimaldot]{babel}
\usepackage{amsmath,amssymb}
\usepackage{newpxtext,newpxmath}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc}
""" + "".join(rf"\definecolor{{{k}}}{{HTML}}{{{v}}}" + "\n" for k, v in HEX.items()) + r"""
\newcommand{\no}[1]{{\color{orange}#1}}
\newcommand{\nod}[1]{{\color{oscuro}#1}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.45em}
\begin{document}
\hyphenpenalty=10000\exhyphenpenalty=10000
"""
DPI = 600


def latex_png(cuerpo):
    """Compila `cuerpo` (dentro de document) y devuelve la ruta del PNG (caché)."""
    src = PREAMBULO + cuerpo + "\n\\end{document}\n"
    h = hashlib.sha1(src.encode("utf-8")).hexdigest()[:14]
    png = TEXDIR / f"{h}.png"
    if png.exists():
        return png
    tex = TEXDIR / f"{h}.tex"
    tex.write_text(src, encoding="utf-8")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        tex.name], cwd=TEXDIR, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise SystemExit(f"pdflatex falló en {tex}")
    subprocess.run(["pdftocairo", "-png", "-transp", "-singlefile", "-r",
                    str(DPI), f"{h}.pdf", h], cwd=TEXDIR, check=True)
    for ext in (".aux", ".log"):
        (TEXDIR / f"{h}{ext}").unlink(missing_ok=True)
    return png


def bloque(ancho_in, contenido, size=B, salto=None):
    """Párrafo(s) de texto con matemáticas, en una caja de ancho fijo."""
    salto = salto or round(size * 1.24, 1)
    return (rf"\begin{{minipage}}{{{ancho_in * CM:.3f}cm}}"
            rf"\fontsize{{{size}}}{{{salto}}}\selectfont\raggedright "
            + contenido + r"\end{minipage}")


def picture(slide, png, x, y, w=None, name=None, dpi=DPI):
    """Inserta el PNG a escala 1:1 (o al ancho w). Devuelve el alto en pulgadas."""
    px_w, px_h = Image.open(png).size
    nat_w, nat_h = px_w / dpi, px_h / dpi
    if w is None or nat_w <= w:
        w_, h_ = nat_w, nat_h
    else:
        w_, h_ = w, nat_h * w / nat_w
    pic = slide.shapes.add_picture(str(png), I(x), I(y), I(w_), I(h_))
    if name:
        pic.name = name
    return h_


def tex(slide, x, y, w, contenido, size=B, name=None, centrar=False):
    png = latex_png(bloque(w, contenido, size))
    return picture(slide, png, x, y, w, name=name)


def tikz(slide, x, y, w, cuerpo, size=20, name=None):
    png = latex_png(rf"\fontsize{{{size}}}{{{round(size * 1.2, 1)}}}\selectfont"
                    + cuerpo)
    px_w, _ = Image.open(png).size
    dx = max(0.0, (w - px_w / DPI) / 2)       # centrado si es más angosto
    return picture(slide, png, x + dx, y, w, name=name)


# ---------------------------------------------------------------- datos MC
def leer(carpeta, patron):
    filas = []
    for f in sorted(glob.glob(str(Path(carpeta) / patron))):
        with open(f, newline="", encoding="utf-8") as fh:
            filas += list(csv.DictReader(fh))
    return filas


def promedio_params(patron):
    """Promedio sobre réplicas del mejor arranque.  Si una réplica se cortó y se
    re-estimó (fila repetida), vale la última."""
    ultimo = {}
    verdad = {}
    for r in leer(ARGS.mc, patron):
        if r["mejor"] != "True":
            continue
        ultimo[(r["rep"], r["estimador"], r["parametro"])] = float(r["estimado"])
        verdad[r["parametro"]] = float(r["verdad"])
    acc = defaultdict(list)
    for (_, est, par), v in ultimo.items():
        acc[(est, par)].append(v)
    N_REPS.update({est: len({k[0] for k in ultimo if k[1] == est})
                   for est in {k[1] for k in ultimo}})
    return {k: sum(v) / len(v) for k, v in acc.items()}, verdad


N_REPS = {}


EST, VERDAD = promedio_params("parametros_reps*.csv")
GEST, GVERDAD = promedio_params("gill_parametros_reps*.csv")
EST.update(GEST)
print("réplicas por estimador:", N_REPS)


def sesgo(est, p):
    return 100 * (est / VERDAD[p] - 1)


def precios_por_edad():
    """Precio promedio (ponderado por la masa de coches q, sobre marcas) por edad,
    en el mercado base, promediado sobre réplicas, para cada estimador."""
    carpeta = Path(ARGS.precios or ARGS.mc)
    if not (carpeta / "por_edad_mc.csv").exists():
        carpeta = MCDIR / "smoke"
        print(f"AVISO: sin por_edad_mc.csv en --precios; uso {carpeta}")
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))  # (est, rep) -> a
    for r in leer(carpeta, "por_edad_mc.csv"):
        if r["regimen"] != "1":
            continue
        c = acc[(r["estimador"], r["rep"])][int(r["a"])]
        c[0] += float(r["q"]) * float(r["P_transaccion"])
        c[1] += float(r["q"])
    for r in leer(carpeta, "gill_por_edad_mc.csv"):
        if r["estimador"] != "gill_completa":
            continue
        c = acc[(r["estimador"], r["rep"])][int(r["a"])]
        c[0] += float(r["q"]) * float(r["P"])
        c[1] += float(r["q"])
    curvas = defaultdict(lambda: defaultdict(list))
    for (est, _), por_a in acc.items():
        for a, (pq, q) in por_a.items():
            curvas[est][a].append(pq / q)
    return {est: {a: sum(v) / len(v) for a, v in sorted(d.items())}
            for est, d in curvas.items()}


PRECIOS = precios_por_edad()

# ---------------------------------------------------------------- gráficas
plt.rcParams.update({
    "font.family": FONT, "mathtext.fontset": "custom",
    "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "axes.edgecolor": "#" + HEX["rule"], "axes.labelcolor": "#" + HEX["text"],
    "xtick.color": "#" + HEX["muted"], "ytick.color": "#" + HEX["muted"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E2E7EE", "grid.linewidth": 1.0,
    "legend.frameon": False, "savefig.transparent": True,
})
C = {k: "#" + v for k, v in HEX.items()}
COL_EST = {"verdad": C["text"], "gill_completa": C["orange"],
           "diseno_1": C["sky"], "diseno_2": "#5E86C6", "diseno_3": C["navy"]}
NOM_EST = {"verdad": "Verdadero", "gill_completa": "Gillingham (sin reparación)",
           "diseno_1": "Diseño 1", "diseno_2": "Diseño 2", "diseno_3": "Diseño 3"}


def fig_mezcla(w, h):
    d_, k_, sg = 0.075, 0.25, 0.15
    pesos = [("Reparar barato", 0.45, C["sky"]),
             ("Precio medio", 0.30, C["blue"]),
             ("Reparar caro", 0.15, C["navy"])]
    zs = [-0.75 + 0.005 * i for i in range(281)]

    def npdf(z, m):
        return math.exp(-0.5 * ((z - m) / sg) ** 2) / (sg * math.sqrt(2 * math.pi))

    fig, ax = plt.subplots(figsize=(w, h))
    for nom, p, col in pesos:
        ax.plot(zs, [p * npdf(z, d_ - k_) + (1 - p) * npdf(z, d_) for z in zs],
                color=col, lw=4.5, label=nom)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_xlim(-0.75, 0.6)
    ax.tick_params(labelsize=XS)
    ax.set_xlabel(r"desgaste al año siguiente, $w_{t+1}$   (hoy $w_t = 0$)",
                  fontsize=XS + 1)
    ax.legend(fontsize=XS, loc="upper left", handlelength=1.6)
    fig.tight_layout(pad=0.3)
    out = FIGDIR / "mezcla.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig_precios(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    for est in ("gill_completa", "diseno_1", "diseno_2", "diseno_3", "verdad"):
        if est not in PRECIOS:
            continue
        a, p = zip(*PRECIOS[est].items())
        kw = dict(color=COL_EST[est], label=NOM_EST[est], lw=3.0, zorder=3)
        if est == "verdad":     # banda ancha detrás: los diseños caen encima
            kw.update(color="#C3CDDA", lw=12, zorder=1, solid_capstyle="round")
        ax.plot(a, p, **kw)
    h_, l_ = ax.get_legend_handles_labels()
    orden = [l_.index(n) for n in [NOM_EST[e] for e in
             ("verdad", "diseno_1", "diseno_2", "diseno_3", "gill_completa")]
             if n in l_]
    h_, l_ = [h_[i] for i in orden], [l_[i] for i in orden]
    ax.tick_params(labelsize=XS)
    ax.set_xlabel("edad del coche (años)", fontsize=S)
    ax.set_ylabel("miles de DKK", fontsize=S)
    ax.grid(axis="x", visible=False)
    ax.set_title("Precio de los usados por edad", fontsize=S + 2,
                 color=C["navy"], loc="left", fontweight="bold")
    ax.legend(h_, l_, fontsize=XS - 1, loc="upper right", handlelength=1.8)
    ax.text(0.97, 0.52, "los diseños 1–3 caen\nsobre el verdadero",
            transform=ax.transAxes, ha="right", va="top", fontsize=XS - 1,
            style="italic", color=C["muted"])
    fig.tight_layout(pad=0.3)
    out = FIGDIR / "precios.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig_sesgo(w, h):
    keys = [("mu_t0", r"$\mu$  sensibilidad al precio"),
            ("u0_t0_j0", r"$u_0$  utilidad base"),
            ("u1_t0_j0", r"$u_1$  pérdida por edad"),
            ("acc_age_j0", r"$\alpha_1$  avería por edad"),
            ("tc_buy_t0", r"$T_b$  costo de comprar")]
    ests = ["gill_completa", "diseno_1", "diseno_2", "diseno_3"]
    fig, ax = plt.subplots(figsize=(w, h))
    bh = 0.19
    for k, est in enumerate(ests):
        ys = [i + (k - 1.5) * (bh + 0.015) for i in range(len(keys))]
        xs = [sesgo(EST[(est, p)], p) for p, _ in keys]
        ax.barh(ys, xs, height=bh, color=COL_EST[est], label=NOM_EST[est])
        if est == "gill_completa":
            for yy, xx in zip(ys, xs):
                ax.text(xx + (-1.5 if xx < 0 else 1.5), yy, f"{xx:+.0f}%",
                        va="center", ha="right" if xx < 0 else "left",
                        fontsize=XS - 2, color=C["text"])
    ax.axvline(0, color=C["muted"], lw=1.2)
    peor = max(abs(sesgo(EST[(e, p)], p)) for e in ests[1:] for p, _ in keys)
    ax.text(1.2, len(keys) - 0.62, f"diseños 1–3:\n|sesgo| < {math.ceil(peor)}%",
            ha="left", va="center", fontsize=XS - 1, style="italic",
            color=C["navy"])
    ax.set_yticks(range(len(keys)), [lab for _, lab in keys], fontsize=XS)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=XS - 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    lo = min(sesgo(EST[(e, p)], p) for e in ests for p, _ in keys)
    ax.set_xlim(min(-10, lo * 1.25), 20)
    ax.set_title("Sesgo, % respecto al valor verdadero", fontsize=S + 2,
                 color=C["navy"], loc="left", fontweight="bold")
    ax.legend(fontsize=XS - 1, loc="upper center", handlelength=1.2, ncol=2,
              bbox_to_anchor=(0.4, -0.1))
    fig.tight_layout(pad=0.3)
    out = FIGDIR / "sesgo.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- helpers pptx
def fill_tf(tf, paras, size=B, color=TEXT, align=PP_ALIGN.LEFT, after=6):
    """Cada párrafo: str o dict(text, size, color, b, i, align, after).
    Dentro del texto, **negrita** y __itálica__."""
    import re
    first = True
    for item in paras:
        if isinstance(item, str):
            item = {"text": item}
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = item.get("align", align)
        p.space_after = Pt(item.get("after", after))
        for piece in re.split(r"(\*\*.+?\*\*|__.+?__)", item["text"]):
            if not piece:
                continue
            r = p.add_run()
            b, i = item.get("b", False), item.get("i", False)
            if piece.startswith("**"):
                piece, b = piece[2:-2], True
            elif piece.startswith("__"):
                piece, i = piece[2:-2], True
            r.text = piece
            f = r.font
            f.size = Pt(item.get("size", size))
            f.name = FONT
            f.bold, f.italic = b, i
            f.color.rgb = item.get("color", color)


def textbox(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, name=None, **kw):
    s = slide.shapes.add_textbox(I(x), I(y), I(w), I(h))
    tf = s.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", 0)
    fill_tf(tf, paras, **kw)
    if name:
        s.name = name
    return s


def rect(slide, x, y, w, h, fill=None, line=None, lw=1.25, name=None, r=0.0):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if r else MSO_SHAPE.RECTANGLE,
        I(x), I(y), I(w), I(h))
    if r:
        shp.adjustments[0] = min(0.5, r / min(w, h))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(lw)
    shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def hline(slide, x1, y1, x2, y2, color=RULE, w=1.25):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, I(x1), I(y1),
                                   I(x2), I(y2))
    c.line.color.rgb = color
    c.line.width = Pt(w)
    return c


def section(slide, x, y, num, title, w=COLW):
    """Barra de sección: número y título en blanco sobre azul marino."""
    rect(slide, x, y, w, 1.0, fill=NAVY, name=f"Sección {num}")
    textbox(slide, x + 0.3, y, w - 0.6, 1.0, [
        {"text": f"{num}.  {title}", "size": 36, "b": True, "color": WHITE,
         "after": 0}], anchor=MSO_ANCHOR.MIDDLE)
    return y + 1.35


def mini_table(slide, x, y, w, rows, colw, rowh, size=S, name=None,
               aligns=None, styler=None):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, I(x), I(y), I(w), I(rowh * nr))
    if name:
        gt.name = name
    tbl = gt.table
    tblPr = tbl._tbl.tblPr
    tblPr.set("bandRow", "0")
    tblPr.set("firstRow", "0")
    for j, cw in enumerate(colw):
        tbl.columns[j].width = I(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = I(rowh)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = I(0.08)
            cell.margin_top = cell.margin_bottom = I(0.03)
            tf = cell.text_frame
            tf.word_wrap = True
            st = {"size": size, "color": TEXT, "b": False, "fill": WHITE}
            if i == 0:
                st.update(b=True, color=NAVY)
            if styler:
                styler(i, j, val, st)
            fill_tf(tf, [{"text": val, "size": st["size"], "color": st["color"],
                          "b": st["b"], "after": 0,
                          "align": aligns[j] if aligns else PP_ALIGN.LEFT}])
            tcPr = cell._tc.get_or_add_tcPr()
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tcPr, qn(tag))
                if tag == "a:lnB" or (tag == "a:lnT" and i == 0):
                    ln.set("w", str(int(Pt(1.75 if i == 0 else 0.75))))
                    sf = etree.SubElement(ln, qn("a:solidFill"))
                    etree.SubElement(sf, qn("a:srgbClr")).set(
                        "val", str(NAVY if i == 0 else RULE))
                else:
                    ln.set("w", "0")
                    etree.SubElement(ln, qn("a:noFill"))
            sf = etree.SubElement(tcPr, qn("a:solidFill"))
            etree.SubElement(sf, qn("a:srgbClr")).set("val", str(st["fill"]))
    return gt


# ================================================================ TEXTOS LaTeX
# Árbol de decisiones de un hogar con coche (sección 4).
TEX_ARBOL = r"""
\begin{tikzpicture}[
  n/.style={draw=navy!45, fill=pale, rounded corners=4pt, align=center,
            inner sep=4pt, minimum height=1.55cm, text width=#1},
  n/.default=3.95cm,
  oc/.style={draw=orange, fill=cream, text=orange, line width=1.4pt,
             font=\bfseries},
  e/.style={draw=muted, line width=1.1pt}]
\node[n=11.5cm, fill=navy, draw=navy, text=white, font=\bfseries] (root) at (0,0)
     {Hogar con coche $x=(j,a,w)$};
\node[n=4.6cm] (des) at (-8.9,-2.7) {Deshacerse};
\node[n=4.6cm] (man) at (0,-2.7) {Mantener};
\node[n=4.6cm] (int) at (8.9,-2.7) {Intercambiar};
\node[n] (v) at (-11.05,-5.4) {Vender\\[-2pt] $D=0$};
\node[n] (c) at (-6.75,-5.4) {Chatarrizar\\[-2pt] $D=1$};
\node[n,oc] (r1) at (-2.15,-5.4) {Reparar\\[-2pt] $r=1$};
\node[n,oc] (r0) at (2.15,-5.4) {No reparar\\[-2pt] $r=0$};
\node[n=8.3cm] (vc) at (8.9,-5.4) {Vende o chatarriza el suyo\\[-2pt]
     y compra $h=(i,b,y)$};
\node[n,oc] (r1b) at (6.75,-8.3) {Reparar\\[-2pt] $r=1$};
\node[n,oc] (r0b) at (11.05,-8.3) {No reparar\\[-2pt] $r=0$};
\foreach \p/\h in {root/des,root/man,root/int,des/v,des/c,man/r1,man/r0,
                   int/vc,vc/r1b,vc/r0b}
  \draw[e] (\p.south) -- ++(0,-0.4) -| (\h.north);
\node[align=left, text width=13cm, anchor=west] at (-13.1,-8.3)
  {\textcolor{orange}{\textbf{En naranja:}} lo que el econometrista no
   observa. Los coches nuevos y los de la última edad no se reparan.};
\end{tikzpicture}
"""

TEX_MODELO = r"""
Para un estado $x=(j,a,w)$ \,---marca, edad y desgaste del coche---\, si con la
decisión $d$ un hogar de tipo $\tau$ se queda con el coche $h=(i,b,y)$, su utilidad
del año es
\[ U_\tau(d)=u_\tau(h)-\mu_\tau\Big[\text{gastos}-\text{ingresos}\Big]. \]
El valor de cada decisión suma el futuro: si el coche se avería pasa a la edad
terminal $\bar a$; si sobrevive, envejece y su desgaste cambia,
\[ \begin{aligned} v(x;d)=U_\tau(d)+\beta\Big[\,&s(h)\,EV(i,\bar a)\\
   &+\big(1-s(h)\big)\!\int\! f(w'\mid h,r)\,EV(i,b{+}1,w')\,dw'\Big]. \end{aligned} \]
Con choques $\epsilon(d)$ de valor extremo generalizado (GEV), la ecuación de Bellman
\[ V(x,\epsilon)=\max_{d\in C(x)}\Big\{v(x;d)+\epsilon(d)\Big\},\qquad
   EV(x)=\mathbb{E}_\epsilon\Big[V(x,\epsilon)\Big] \]
da probabilidades de elección $\Pi(d\mid x;P,R)$ en forma cerrada.

\textbf{Equilibrio.} Con esas probabilidades se suman demanda y oferta de cada
usado; los precios $P$ vacían el mercado y la distribución $q$ es estacionaria:
\[ \underbrace{D_{jaw}(P)}_{\text{demanda}}=\underbrace{S_{jaw}(P)}_{\text{oferta}}
   \quad\forall\,(j,a,w),\qquad q=q\,M(P,R). \]
"""

TEX_REPARACION = r"""
El \textbf{\textcolor{blue}{desgaste}} del coche evoluciona como
{\fontsize{34}{42}\selectfont
\[ \textcolor{blue}{\boldsymbol{w_{t+1}}}=m\big(j,\,\nod{r_t},\,w_t\big)+\eta_{t+1},
   \qquad \eta\sim N\big(0,\sigma_\eta^2\big) \]}%
y cada año el coche se avería con probabilidad
{\fontsize{34}{42}\selectfont
\[ s(j,a,w). \]}%
Reparar ($\nod{r_t}=1$) baja el desgaste y, con él, \textbf{cambia la probabilidad de
avería}.
"""

TEX_MEZCLA_TXT = r"""
Distribución \textbf{observada} del desgaste al año siguiente en tres mercados.
Donde reparar es barato, más hogares reparan y la masa se corre a la izquierda.
"""

# Árbol de la verosimilitud: una observación entre t y t+1 (de presentacion_vero).
TEX_ARBOL_VERO = r"""
\begin{tikzpicture}[
  n/.style={draw=navy!45, fill=pale, rounded corners=4pt, align=center,
            text width=3.1cm, inner sep=5pt, minimum height=1.45cm},
  ob/.style={draw=navy, fill=navy, text=white, text width=3.9cm},
  oc/.style={draw=orange, fill=cream, text=orange, line width=1.4pt,
             font=\bfseries, text width=3.4cm},
  e/.style={-{Stealth[length=9pt]}, draw=muted, line width=1.1pt},
  lab/.style={font=\fontsize{19}{21}\selectfont, inner sep=3pt, text=text}]
\node[n, ob] (h) at (0,0) {$x=(j,a,w)$\\[-1pt] se observa};
\node[n] (acc) at (5.9,2.7) {avería};
\node[n] (sob) at (5.9,-2.0) {sobrevive};
\node[n, oc] (r1) at (11.6,0.0) {reparó};
\node[n, oc] (r0) at (11.6,-4.0) {no reparó};
\node[n, ob] (w) at (18.2,-2.0) {$(j,a{+}1,w')$\\[-1pt] se observa};
\node[n, ob] (d) at (18.2,-6.6) {decisión $e$\\[-1pt] y coche $\delta'$};
\draw[e] (h) -- node[lab, above left]{$s$} (acc);
\draw[e] (h) -- node[lab, below left]{$1-s$} (sob);
\draw[e] (sob) -- node[lab, above left]{$\Pi(r{=}1\mid h)$} (r1);
\draw[e] (sob) -- node[lab, below left]{$\Pi(r{=}0\mid h)$} (r0);
\draw[e] (r1) -- node[lab, above right]{$f(w'\mid h,1)$} (w);
\draw[e] (r0) -- node[lab, below right]{$f(w'\mid h,0)$} (w);
\draw[e] (w) -- node[lab, right]{$\Pi(e,\delta'\mid j,a{+}1,w')$} (d);
\draw[e] (acc.east) -- node[lab, above]{$\Pi(\delta'\mid j,\bar a)$} (23.6,2.7)
     |- (d.east);
\begin{scope}[on background layer]
  \node[draw=orange, line width=2.4pt, rounded corners=10pt, fill=cream!45,
        inner xsep=0.45cm, inner ysep=0.5cm, fit=(r1)(r0)] (box) {};
\end{scope}
\node[below=4pt of box, text=orange, font=\bfseries, align=center]
     {mezcla $Z(w'\mid h)$};
\end{tikzpicture}
"""

TEX_VERO = r"""
Un hogar empieza en $x=(j,a,w)$; al año siguiente vemos qué decidió ($e$) y con qué
coche quedó ($\delta'$). Su contribución a la verosimilitud suma los caminos que no
vemos:
\[ \begin{aligned} \ell(\theta)=\;&s(x)\,\Pi\big(\delta'\mid j,\bar a\big)\,\mathbf{1}[D{=}1]\\
   &+\big(1-s(x)\big)\sum_{w'}\no{Z(w'\mid x)}\;\Pi\big(e,\delta'\mid j,a{+}1,w'\big). \end{aligned} \]
La reparación no observada entra como una \textbf{mezcla} de las dos campanas:
\[ \no{Z(w'\mid x)}=\sum_{r\in\{0,1\}}\Pi\big(r\mid x;\,P_m,R_m\big)\;
   f\big(w'\mid x,r\big). \]
Los pesos $\Pi(r\mid x)$ cambian con el precio de reparación $R_m$ de cada mercado;
las densidades $f$, no.
"""

# ================================================================ PÓSTER
prs = Presentation()
prs.slide_width = I(W_IN)
prs.slide_height = I(H_IN)
sl = prs.slides.add_slide(prs.slide_layouts[6])
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE

# ---- encabezado
rect(sl, 0, 0, W_IN, 4.15, fill=NAVY, name="Encabezado")
textbox(sl, MARGIN, 0.5, W_IN - 2 * MARGIN, 3.5, [
    {"text": "¿Qué tanto importa la reparación?", "size": 80, "b": True,
     "color": WHITE, "after": 0, "align": PP_ALIGN.CENTER},
    {"text": "Decisiones de reparación no observadas en el mercado de coches",
     "size": 44, "color": WHITE, "after": 10, "align": PP_ALIGN.CENTER},
    {"text": "Rodrigo Antonio Aldrette Salas   ·   Asesor: Edwin Alexander Muñoz "
             "Rodríguez   ·   El Colegio de México   ·   Tesis de licenciatura, 2026",
     "size": 28, "color": SOFT, "after": 0, "align": PP_ALIGN.CENTER},
], name="Título")
rect(sl, 0, 4.15, W_IN, 0.14, fill=ORANGE, name="Filete")
textbox(sl, MARGIN + 1.5, 4.5, W_IN - 2 * MARGIN - 3.0, 1.4, [
    {"text": "La reparación determina cuánto dura un coche y cuánto vale como usado: "
             "tiene un papel clave en el mercado de los bienes durables. Ignorarla "
             "no es neutral: un modelo sin reparación, estimado con hogares que sí "
             "reparan, da parámetros y precios sesgados.",
     "size": 32, "i": True, "color": NAVY, "after": 0,
     "align": PP_ALIGN.CENTER}], anchor=MSO_ANCHOR.MIDDLE, name="Idea central")

COL_END = []

# ================================================================ COLUMNA 1
x, y = COLX[0], Y0
y = section(sl, x, y, 1, "La pregunta")
textbox(sl, x, y, COLW, 3.0, [
    {"text": "¿Cómo podemos modelar la reparación en un mercado de coches? Y, "
             "¿podemos estimar esta decisión comúnmente no observada?",
     "size": 32, "b": True, "color": NAVY, "after": 0},
])
y += 2.5
why = [("Lo que observamos", "Qué coche tiene cada hogar, su desgaste y si lo "
                             "mantiene, vende o chatarriza."),
       ("Lo que **no** observamos", "Quién **reparó**, ni si un coche salió por "
                                    "avería o por decisión del dueño."),
       ("Por eso estimamos", "**Un modelo estructural donde reparar es una "
                             "decisión endógena del hogar.**")]
for k, (t, d) in enumerate(why):
    textbox(sl, x, y, 3.0, 1.8, [
        {"text": t, "color": BLUE if k == 2 else MUTED, "after": 0}])
    textbox(sl, x + 3.15, y, COLW - 3.15, 1.8, [{"text": d, "after": 0}])
    y += 1.75
y += 0.4

y = section(sl, x, y, 2, "Mi aportación")
aport = [
    ("Teoría", "Modelo la reparación como **decisión endógena** del hogar dentro "
               "de un equilibrio de coches nuevos y usados."),
    ("Monte Carlo", "Simulo un mercado donde los hogares sí reparan y mido cuánta "
                    "información se pierde al no ver la reparación y el **sesgo** de "
                    "estimar un modelo sin ella."),
    ("Econometría", "Una verosimilitud que estima el modelo **sin ver la "
                    "reparación** ni el motivo de salida de los coches, llevando a "
                    "Hu & Xin a un modelo de equilibrio."),
]
for k, (t, d) in enumerate(aport):
    textbox(sl, x, y, 3.0, 2.3, [{"text": t, "b": True, "color": BLUE,
                                  "after": 0}])
    textbox(sl, x + 3.15, y, COLW - 3.15, 2.3, [{"text": d, "after": 0}])
    y += 2.8
    if k < 2:
        hline(sl, x, y - 0.15, x + COLW, y - 0.15)
y += 0.3

y = section(sl, x, y, 3, "Sobre los hombros de…")
lit = [
    ("Rust (1987)", "Cuándo reemplazar el motor de un autobús. Punto fijo "
                    "anidado: resolver Bellman dentro de la verosimilitud. "
                    "__La decisión sí se observaba.__"),
    ("Gillingham et al. (2022)", "Equilibrio de coches nuevos y usados en "
                                 "Dinamarca; los precios de los usados salen del "
                                 "modelo. __No hay reparación.__"),
    ("Hu & Xin (2024)", "Identificación de decisiones no observadas con un "
                        "estado continuo y una variable excluida. __Sin "
                        "equilibrio de mercado.__"),
]
for k, (t, d) in enumerate(lit):
    textbox(sl, x, y, 3.0, 1.9, [{"text": t, "size": S, "b": True,
                                  "color": NAVY, "after": 0}])
    textbox(sl, x + 3.15, y, COLW - 3.15, 1.9, [{"text": d, "size": S,
                                                  "after": 0}])
    y += 2.2
    if k < 2:
        hline(sl, x, y - 0.15, x + COLW, y - 0.15)
COL_END.append(y)

# ================================================================ COLUMNA 2
x, y = COLX[1], Y0
y = section(sl, x, y, 4, "El modelo")
textbox(sl, x, y, COLW, 2.0, [
    {"text": "Hogares que viven para siempre deciden cada año qué coche tener. "
             "El equilibrio es **estacionario** y los precios de los coches usados "
             "son **endógenos**: se ajustan hasta vaciar el mercado.",
     "after": 0}])
y += 1.75
y += tikz(sl, x, y, COLW, TEX_ARBOL, size=20, name="Árbol de decisiones") + 0.45
y += tex(sl, x, y, COLW, TEX_MODELO, name="Utilidad y Bellman") + 0.5

y = section(sl, x, y, 5, "La reparación")
y += tex(sl, x, y, COLW, TEX_REPARACION, name="Desgaste y avería") + 0.3
MZH = 3.4
y += picture(sl, fig_mezcla(COLW, MZH), x, y, COLW, name="Gráfica mezcla",
             dpi=300) + 0.15
y += tex(sl, x, y, COLW, TEX_MEZCLA_TXT, size=XS, name="Variable excluida")
COL_END.append(y)

# ================================================================ COLUMNA 3
FOOT = H_IN - 1.2
x, y = COLX[2], Y0
y = section(sl, x, y, 6, "La verosimilitud")
y += tikz(sl, x, y, COLW, TEX_ARBOL_VERO, size=20,
          name="Árbol de la verosimilitud") + 0.5
y += tex(sl, x, y, COLW, TEX_VERO, name="Verosimilitud") + 0.3

textbox(sl, x, y, COLW, 0.5, [{"text": "Estimación por punto fijo anidado",
                               "size": B, "b": True, "color": BLUE, "after": 0}])
y += 0.62
steps = [("1 · Proponer θ", "parámetros"),
         ("2 · Bellman", "qué hace cada hogar"),
         ("3 · Precios P(θ)", "equilibrio por mercado"),
         ("4 · LL(θ)", "con la mezcla")]
AW = 0.3
sw = (COLW - 3 * AW) / 4
for k, (t, d) in enumerate(steps):
    bx = x + k * (sw + AW)
    shp = rect(sl, bx, y, sw, 1.75, fill=PALE, r=0.12)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", I(0.08))
    fill_tf(tf, [{"text": t, "size": 22, "b": True, "color": BLUE, "after": 1,
                  "align": PP_ALIGN.CENTER},
                 {"text": d, "size": 20, "color": TEXT, "after": 0,
                  "align": PP_ALIGN.CENTER}])
    if k < 3:
        textbox(sl, bx + sw, y + 0.55, AW, 0.6, [
            {"text": "→", "size": 24, "color": MUTED, "align": PP_ALIGN.CENTER,
             "after": 0}])
y += 1.83
textbox(sl, x, y, COLW, 0.5, [
    {"text": "↻ se repite hasta maximizar LL(θ); los precios nunca se observan, "
             "salen del equilibrio (Rust, 1987; Gillingham et al., 2022).",
     "size": 20, "color": MUTED, "after": 0}])
y += 1.05

# ---- resultados, dentro de un cuadro azul de esquinas suaves
PAD = 0.3
BOX_Y = y
rect(sl, x, BOX_Y, COLW, FOOT - 0.3 - BOX_Y, fill=None, line=BLUE, lw=3.0,
     r=0.35, name="Cuadro resultados")
xi, wi = x + PAD, COLW - 2 * PAD
y = section(sl, xi, y + PAD, 8, "Resultados", w=wi)
textbox(sl, xi, y, wi, 4.4, [
    {"text": "Los **parámetros estructurales nuevos** (el efecto de reparar, el "
             "desgaste anual y su choque, la escala de reparar y la utilidad del "
             "desgaste) **se estiman sin sesgo** en los tres diseños, aun sin ver "
             "la reparación ni el motivo de salida: el sesgo promedio es menor a "
             "1% en todos los parámetros. Ignorar la reparación, como Gillingham, "
             "subestima la sensibilidad al precio en ~40% y sesga la utilidad de "
             "tener coche, y con ellas los precios de los usados.", "after": 0}])
y += 4.5
CH_H = (FOOT - 0.3 - PAD - y - 0.3) / 2
picture(sl, fig_precios(wi, CH_H), xi, y, wi, name="Gráfica precios", dpi=300)
y += CH_H + 0.3
picture(sl, fig_sesgo(wi, CH_H), xi, y, wi, name="Gráfica sesgo", dpi=300)
y += CH_H + PAD

# ================================================================ BANDA INFERIOR: MONTE CARLO
TOP = max(COL_END) + 0.6
COL_END.append(y)
WIDE = 2 * COLW + GAP
x, y = MARGIN, TOP
y = section(sl, x, y, 7, "Monte Carlo", w=WIDE)
LW = 5.6                      # subcolumna izquierda: cifras y explicación
RX, RW = x + LW + 0.7, WIDE - LW - 0.7
yl = y
for n, lab in [("1,000", "réplicas"),
               ("3", "mercados con distinto precio de reparación"),
               ("10,000", "hogares × 4 años por mercado")]:
    textbox(sl, x, yl, LW, 1.6, [
        {"text": n, "size": 50, "b": True, "color": ORANGE, "after": 0},
        {"text": lab, "size": S, "color": MUTED, "after": 0}])
    yl += 1.75
textbox(sl, x, yl + 0.15, LW, 3.0, [
    {"text": "Simulamos un panel completo y cada estimador **tapa columnas** del "
             "mismo panel: las diferencias entre ellos son solo de información.",
     "after": 0}])

cols = ["Hogar", "Rico", "Precio de\nreparación", "Marca · edad", "Desgaste",
        "Decisión", "Motivo", "¿Reparó?"]
cwid = [0.95, 0.75, 1.4, 1.45, 1.15, 2.3, 1.05, 1.15]
cwid = [c * RW / sum(cwid) for c in cwid]
textbox(sl, RX, y, RW, 0.55, [{"text": "La base: una fila por hogar y año",
                               "b": True, "color": BLUE, "after": 0}])
y += 0.65
datos = [cols,
         ["17", "no", "bajo", "LB · 5", "0.30", "mantiene", "—", "sí"],
         ["42", "sí", "alto", "HB · 9", "0.85", "chatarriza y compra", "avería",
          "no"]]
RH = 0.7
mini_table(sl, RX, y, RW, datos, cwid, RH, size=S, name="Base de datos",
           aligns=[PP_ALIGN.CENTER] * 8)
y += RH * 3 + 0.6   # la primera fila (2 líneas) crece un poco

textbox(sl, RX, y, RW, 0.55, [{"text": "Qué ve cada estimador", "b": True,
                               "color": BLUE, "after": 0}])
y += 0.65
tapa = {"Gillingham": {"Desgaste", "¿Reparó?"},
        "Diseño 1": set(), "Diseño 2": {"¿Reparó?"},
        "Diseño 3": {"¿Reparó?", "Motivo"}}
mask = [["", *cols[1:]]]
for dname, hid in tapa.items():
    mask.append([dname] + ["" if c in hid else "✓" for c in cols[1:]])
mw = [cwid[0] + 1.0] + [(RW - cwid[0] - 1.0) * c / sum(cwid[1:])
                        for c in cwid[1:]]


def estilo_mask(i, j, val, st):
    if i == 0:
        st["size"] = XS
        return
    if j == 0:
        st.update(b=True, color=ORANGE if i == 1 else NAVY)
        return
    if val == "":
        st["fill"] = REDACT
    else:
        st.update(color=BLUE)


MH = 0.66
mini_table(sl, RX, y, RW, mask, mw, MH, size=S, name="Máscaras",
           aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 7, styler=estilo_mask)
y += MH * 5 + 0.45
rect(sl, RX, y + 0.1, 0.36, 0.26, fill=REDACT)
textbox(sl, RX + 0.5, y, RW - 0.5, 0.9, [
    {"text": "tapado. Gillingham es su modelo, sin desgaste ni reparación. Nadie "
             "ve los precios de los usados.", "size": XS, "color": MUTED,
     "after": 0}])
y += 0.9
COL_END.append(y)

# ---- referencias (pie)
hline(sl, MARGIN, FOOT, W_IN - MARGIN, FOOT)
textbox(sl, MARGIN, FOOT + 0.12, W_IN - 2 * MARGIN, 1.0, [
    {"text": "Gillingham, K., Iskhakov, F., Munk-Nielsen, A., Rust, J. y "
             "Schjerning, B. (2022). Equilibrium trade in automobiles. "
             "__Journal of Political Economy__.   ·   Hu, Y. y Xin, Y. (2024). "
             "Identification and estimation of dynamic structural models with "
             "unobserved choices. __Journal of Econometrics__.   ·   Rust, J. "
             "(1987). Optimal replacement of GMC bus engines. __Econometrica__.",
     "size": 17, "color": MUTED, "after": 0}])

print("columnas terminan en:", " ".join(f"{v:.2f}" for v in COL_END),
      f"| banda desde {TOP:.2f}, alto gráficas {CH_H:.2f} (pie en {FOOT:.2f})")
prs.save(ARGS.out)
print("guardado:", ARGS.out)

if not ARGS.sin_png:
    ps = shutil.which("powershell") or "powershell"
    subprocess.run([ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(AQUI / "render.ps1")], check=False)
