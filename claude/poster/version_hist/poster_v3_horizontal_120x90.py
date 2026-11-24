# -*- coding: utf-8 -*-
"""Póster de tesis (A0 horizontal, 4 columnas) con python-pptx.

Uso:     python claude/poster/poster.py  [--mc DIR]
Salida:  claude/poster/poster_tesis.pptx
Render:  powershell -File claude/poster/render.ps1   (PNG con PowerPoint)

Los resultados se leen de los CSV del Monte Carlo (por defecto, la prueba de 2
réplicas `prueba2`). Con las 1000 réplicas basta pasar --mc a su carpeta y
cambiar NOTA_RESULTADOS.
"""
import argparse
import csv
import glob
import math
import re
from collections import defaultdict
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import (XL_CHART_TYPE, XL_LABEL_POSITION,
                             XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION)
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
MC_DEF = RAIZ / "claude/niu/final/output/modelo_tesis/montecarlo/prueba2"
EQ_DEF = RAIZ / "claude/niu/final/output/modelo_tesis/teoria/R_LB"
NOTA_RESULTADOS = "Prueba con 2 réplicas; las 1,000 réplicas están corriendo."

ap = argparse.ArgumentParser()
ap.add_argument("--mc", default=str(MC_DEF))
ap.add_argument("--eq", default=str(EQ_DEF))
ap.add_argument("--out", default=str(AQUI / "poster_tesis.pptx"))
ARGS = ap.parse_args()

# ---------------------------------------------------------------- estilo
NAVY = RGBColor(0x17, 0x24, 0x3A)
BLUE = RGBColor(0x2A, 0x5C, 0xAA)
ORANGE = RGBColor(0xD2, 0x69, 0x1E)
CREAM = RGBColor(0xFC, 0xEF, 0xE3)
PALE = RGBColor(0xEE, 0xF2, 0xF8)
TEXT = RGBColor(0x1E, 0x23, 0x2B)
MUTED = RGBColor(0x5F, 0x68, 0x73)
RULE = RGBColor(0xC8, 0xD0, 0xDA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SOFT = RGBColor(0xC9, 0xD3, 0xE3)
REDACT = RGBColor(0x3C, 0x45, 0x52)

HEAD = "Segoe UI Semibold"
BODY = "Segoe UI"
MATH = "Cambria Math"

B, S, XS = 22, 19, 16

W_IN, H_IN = 120 / 2.54, 90 / 2.54    # 120 x 90 cm
MARGIN, GAP = 1.0, 0.9
NCOL = 4
COLW = (W_IN - 2 * MARGIN - (NCOL - 1) * GAP) / NCOL
COLX = [MARGIN + k * (COLW + GAP) for k in range(NCOL)]
Y0 = 7.0


def I(x):
    return Inches(x)


# ---------------------------------------------------------------- datos MC
def leer(patron):
    filas = []
    for f in sorted(glob.glob(str(Path(ARGS.mc) / patron))):
        with open(f, newline="", encoding="utf-8") as fh:
            filas += list(csv.DictReader(fh))
    return filas


def promedio_params(patron):
    acc = defaultdict(list)
    verdad = {}
    for r in leer(patron):
        if r["mejor"] != "True":
            continue
        acc[(r["estimador"], r["parametro"])].append(float(r["estimado"]))
        verdad[r["parametro"]] = float(r["verdad"])
    return {k: sum(v) / len(v) for k, v in acc.items()}, verdad


# precios de equilibrio por marca y edad (régimen base, zeta = 0)
PRECIOS = defaultdict(dict)       # (modelo, j) -> {a: P}
with open(Path(ARGS.eq) / "por_edad.csv", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["regimen"] == "1":
            PRECIOS[("rep", int(r["j"]))][int(r["a"])] = float(r["P_transaccion"])
            PRECIOS[("gill", int(r["j"]))][int(r["a"])] = float(r["P_gill"])

EST, VERDAD = promedio_params("parametros_reps*.csv")
GEST, GVERDAD = promedio_params("gill_parametros_reps*.csv")
TASA = defaultdict(list)
TASA_V = []
for r in leer("resumen_reps*.csv"):
    TASA[r["estimador"]].append(float(r["reparacion_est"]))
    TASA_V.append(float(r["reparacion_verdad"]))
TASA = {k: sum(v) / len(v) for k, v in TASA.items()}
TASA_V = sum(TASA_V) / len(TASA_V)


def sesgo(est, verdad, p):
    return 100 * (est / verdad[p] - 1)


# ---------------------------------------------------------------- helpers
_MARK = re.compile(r"(\*\*.+?\*\*|\[\[.+?\]\]|__.+?__|\{\{.+?\}\}|<<.+?>>)")
_SUB = re.compile(r"(_\{[^}]*\}|\^\{[^}]*\}|_[^\s{,;)(|·]|\^[^\s{,;)(|·])")


def runs_from(text):
    """**negrita**, [[naranja negrita]], __itálica__, {{fórmula}},
    <<fórmula naranja>>.  Dentro de cualquier texto: x_i, x_{ij}, x^{2}."""
    out = []
    for piece in _MARK.split(text):
        if not piece:
            continue
        if piece.startswith("**"):
            out.append((piece[2:-2], {"b": True}))
        elif piece.startswith("[["):
            out.append((piece[2:-2], {"b": True, "color": ORANGE}))
        elif piece.startswith("__"):
            out.append((piece[2:-2], {"i": True}))
        elif piece.startswith("{{"):
            out.append((piece[2:-2], {"font": MATH}))
        elif piece.startswith("<<"):
            out.append((piece[2:-2], {"font": MATH, "b": True,
                                      "color": ORANGE}))
        else:
            out.append((piece, {}))
    # subíndices y superíndices
    fino = []
    for txt, st in out:
        for sub in _SUB.split(txt):
            if not sub:
                continue
            if sub[0] in "_^" and len(sub) > 1:
                body = sub[2:-1] if sub[1] == "{" else sub[1:]
                fino.append((body, dict(st, base=-25000 if sub[0] == "_"
                                        else 30000)))
            else:
                fino.append((sub, st))
    return fino


def add_runs(p, text, size, color, font, b=False, i=False):
    for txt, st in runs_from(text):
        r = p.add_run()
        r.text = txt
        f = r.font
        f.size = Pt(st.get("size", size))
        f.name = st.get("font", font)
        f.bold = st.get("b", b)
        f.italic = st.get("i", i)
        f.color.rgb = st.get("color", color)
        if "base" in st:
            r._r.get_or_add_rPr().set("baseline", str(st["base"]))


def fill_tf(tf, paras, size=B, color=TEXT, font=BODY, align=PP_ALIGN.LEFT,
            after=6):
    first = True
    for item in paras:
        if isinstance(item, str):
            item = {"text": item}
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = item.get("align", align)
        p.space_after = Pt(item.get("after", after))
        add_runs(p, item["text"], item.get("size", size),
                 item.get("color", color), item.get("font", font),
                 item.get("b", False), item.get("i", False))


def textbox(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, margin=0.0,
            name=None, **kw):
    s = slide.shapes.add_textbox(I(x), I(y), I(w), I(h))
    tf = s.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", I(margin))
    fill_tf(tf, paras, **kw)
    if name:
        s.name = name
    return s


def rect(slide, x, y, w, h, fill=None, line=None, lw=1.25, r=0.08,
         name=None):
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


def put(shp, paras, anchor=MSO_ANCHOR.TOP, margin=0.25, **kw):
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", I(margin))
    fill_tf(tf, paras, **kw)
    return shp


def hline(slide, x1, y1, x2, y2, color=RULE, w=1.25):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, I(x1), I(y1),
                                   I(x2), I(y2))
    c.line.color.rgb = color
    c.line.width = Pt(w)
    return c


def section(slide, x, y, num, title, w=COLW):
    textbox(slide, x, y, 1.0, 0.9, [
        {"text": f"{num:02d}", "size": 30, "color": BLUE, "font": HEAD,
         "after": 0}], anchor=MSO_ANCHOR.BOTTOM)
    textbox(slide, x + 0.95, y, w - 0.95, 0.9, [
        {"text": title, "size": 36, "color": NAVY, "font": HEAD, "after": 0}],
        anchor=MSO_ANCHOR.BOTTOM, name=f"Sección {num:02d}")
    hline(slide, x, y + 1.05, x + w, y + 1.05, color=BLUE, w=2.25)
    return y + 1.35


def label(text, color=BLUE, size=XS):
    """Etiqueta en versalitas, como en el ejemplo (1 · ESTADO)."""
    return {"text": text.upper(), "size": size, "b": True, "color": color,
            "font": HEAD, "after": 2}


def mini_table(slide, x, y, w, rows, colw, rowh, sizes=None, name=None,
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
            p = tf.paragraphs[0]
            p.alignment = (aligns[j] if aligns else PP_ALIGN.LEFT)
            st = {"size": (sizes or [S] * nc)[j], "color": TEXT, "b": False,
                  "font": BODY, "fill": WHITE}
            if i == 0:
                st.update(b=True, font=HEAD, color=NAVY)
            if styler:
                styler(i, j, val, st)
            add_runs(p, val, st["size"], st["color"], st["font"], st["b"])
            # bordes: solo líneas horizontales finas; luego el relleno
            tcPr = cell._tc.get_or_add_tcPr()
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tcPr, qn(tag))
                if tag == "a:lnB" or (tag == "a:lnT" and i == 0):
                    ln.set("w", str(int(Pt(1.5 if i == 0 else 0.75))))
                    sf = etree.SubElement(ln, qn("a:solidFill"))
                    etree.SubElement(sf, qn("a:srgbClr")).set(
                        "val", str(NAVY if i == 0 else RULE))
                else:
                    ln.set("w", "0")
                    etree.SubElement(ln, qn("a:noFill"))
            sf = etree.SubElement(tcPr, qn("a:solidFill"))
            etree.SubElement(sf, qn("a:srgbClr")).set("val", str(st["fill"]))
    return gt


# ---------------------------------------------------------------- póster
prs = Presentation()
prs.slide_width = I(W_IN)
prs.slide_height = I(H_IN)
sl = prs.slides.add_slide(prs.slide_layouts[6])
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE

# ---- encabezado
rect(sl, 0, 0, W_IN, 4.4, fill=NAVY, r=0, name="Encabezado")
textbox(sl, MARGIN, 0.45, W_IN - 2 * MARGIN, 3.8, [
    {"text": "TESIS DE LICENCIATURA · EL COLEGIO DE MÉXICO · ECONOMETRÍA ESTRUCTURAL",
     "size": 20, "color": SOFT, "font": HEAD, "after": 6},
    {"text": "¿Qué tanto importa la reparación?", "size": 72, "color": WHITE,
     "font": HEAD, "after": 0},
    {"text": "Decisiones de reparación no observadas en el mercado de coches",
     "size": 40, "color": WHITE, "font": BODY, "after": 8},
    {"text": "Rodrigo Antonio Aldrette Salas  —  noviembre de 2026",
     "size": 24, "color": SOFT, "after": 0},
], name="Título")

# ---- franja de idea central
rect(sl, 0, 4.4, W_IN, 2.0, fill=CREAM, r=0, name="Idea central")
textbox(sl, MARGIN, 4.4, 2.6, 2.0, [
    {"text": "IDEA", "size": 20, "b": True, "color": ORANGE, "font": HEAD,
     "after": 0},
    {"text": "CENTRAL", "size": 20, "b": True, "color": ORANGE, "font": HEAD,
     "after": 0}], anchor=MSO_ANCHOR.MIDDLE)
textbox(sl, MARGIN + 2.9, 4.4, W_IN - 2 * MARGIN - 2.9, 2.0, [
    {"text": "Aunque nadie registre las reparaciones, el desgaste de los coches "
             "en mercados con distinto precio de reparar revela quién reparó. "
             "Ignorar la reparación sesga los parámetros del modelo.",
     "size": 32, "color": NAVY, "font": HEAD, "after": 0}],
    anchor=MSO_ANCHOR.MIDDLE)

# ================================================================ COLUMNA 1
x, y = COLX[0], Y0
y = section(sl, x, y, 1, "La pregunta")
textbox(sl, x, y, COLW, 3.7, [
    {"text": "Reparar decide cuánto vive un coche y cuánto vale como usado. En "
             "México, un hogar gasta en promedio 19,389 pesos al año en "
             "mantenimiento, pero la reparación casi nunca se observa: buena "
             "parte ocurre en talleres informales.", "size": 26, "after": 10},
    {"text": "**¿Podemos inferir quién reparó a partir de lo que sí vemos? ¿Y "
             "qué tan equivocados estamos si la ignoramos?**", "size": 26,
     "after": 0},
])
y += 3.8
why = [("Los datos muestran", "Qué coche tiene cada hogar, su desgaste y si "
                              "lo mantiene, vende o chatarriza."),
       ("No muestran", "Quién reparó, ni si un coche salió por avería o por "
                       "decisión del dueño."),
       ("Por eso estimamos", "**Un modelo estructural donde reparar es una "
                             "decisión del hogar.**")]
for k, (t, d) in enumerate(why):
    textbox(sl, x, y, 2.9, 1.1, [
        {"text": t, "size": S, "color": BLUE if k == 2 else MUTED,
         "font": HEAD, "after": 0}])
    textbox(sl, x + 3.0, y, COLW - 3.0, 1.1, [
        {"text": d, "size": S, "after": 0}])
    y += 1.25
y += 0.45

y = section(sl, x, y, 2, "Mi aportación")
aport = [
    ("Teoría", "Modelo la reparación como **decisión explícita** del hogar "
               "dentro de un equilibrio de coches nuevos y usados: el precio "
               "de reparar mueve el desgaste y, por ahí, los precios de los "
               "usados."),
    ("Econometría", "Una verosimilitud que estima el modelo **sin ver la "
                    "reparación** ni el motivo de salida de los coches, "
                    "llevando a Hu & Xin a un modelo de equilibrio."),
    ("Monte Carlo", "Mido cuánta información se pierde al no ver la "
                    "reparación y el **sesgo** de estimar un modelo sin ella "
                    "cuando los hogares sí reparan."),
]
for k, (t, d) in enumerate(aport):
    textbox(sl, x, y, 2.9, 1.9, [
        {"text": t, "size": 24, "color": BLUE, "font": HEAD, "after": 0}])
    textbox(sl, x + 3.0, y, COLW - 3.0, 2.2, [{"text": d, "size": 24, "after": 0}])
    y += 2.3
    if k < 2:
        hline(sl, x, y - 0.13, x + COLW, y - 0.13)
y += 0.45

y = section(sl, x, y, 3, "Sobre los hombros de…")
lit = [
    ("Rust (1987)", "Cuándo reemplazar el motor de un autobús. Punto fijo "
                    "anidado: resolver Bellman dentro de la verosimilitud. "
                    "__La decisión sí se observaba.__"),
    ("Gillingham et al. (2022)", "Equilibrio de coches nuevos y usados en "
                                 "Dinamarca; los precios de los usados salen "
                                 "del modelo. __No hay reparación.__"),
    ("Hu & Xin (2024)", "Identificación de decisiones no observadas con un "
                        "estado continuo y una variable excluida. __Sin "
                        "equilibrio de mercado.__"),
]
for k, (t, d) in enumerate(lit):
    textbox(sl, x, y, 2.9, 1.65, [
        {"text": t, "size": 21, "color": NAVY, "font": HEAD, "after": 0}])
    textbox(sl, x + 3.0, y, COLW - 3.0, 1.65, [
        {"text": d, "size": 21, "after": 0}])
    y += 1.75
    if k < 2:
        hline(sl, x, y - 0.12, x + COLW, y - 0.12)
COL_END = [y]

# ================================================================ COLUMNA 2
x, y = COLX[1], Y0
y = section(sl, x, y, 4, "El modelo")

# un año, en cinco pasos (como la tira del ejemplo)
pasos = [("1 · Inicio", "coche (j, a, w)\no ninguno", False),
         ("2 · Mercado", "mantener, vender,\nchatarrizar, comprar", False),
         ("3 · Reparar", "r ∈ {0, 1}\npaga R_{ja}", True),
         ("4 · Avería", "con prob. s;\nsi no, a + 1", False),
         ("5 · Desgaste", "w′ cambia;\nreparar lo baja", False)]
pw = (COLW - 4 * 0.28) / 5
for k, (t, d, hid) in enumerate(pasos):
    bx = x + k * (pw + 0.28)
    put(rect(sl, bx, y, pw, 1.8, fill=CREAM if hid else PALE,
             line=ORANGE if hid else None, lw=1.5),
        [label(t, color=ORANGE if hid else BLUE, size=16)] +
        [{"text": ln, "size": 16, "after": 0} for ln in d.split("\n")],
        margin=0.1)
    if k < 4:
        textbox(sl, bx + pw, y + 0.55, 0.28, 0.6, [
            {"text": "→", "size": 18, "color": MUTED,
             "align": PP_ALIGN.CENTER, "after": 0}])
y += 1.8 + 0.35

eqh = 3.35
put(rect(sl, x, y, COLW, eqh, fill=WHITE, line=RULE, name="Utilidad y Bellman"), [
    label("Utilidad por periodo"),
    {"text": "{{U = u_{0,τj} + u_{1,τj}·a + u_w·w  −  μ_τ·(gastos − ingresos)}}",
     "size": 22, "after": 10},
    label("Bellman: valor de cada decisión"),
    {"text": "{{V(x, ε) = max_{d ∈ C(x)} [ v(x; d) + ε_d ]}}", "size": 22,
     "after": 2},
    {"text": "{{v(x; d) = U(d) + β [ s(h)·EV(j, ā) + (1 − s(h))·E[EV(j, a+1, w′) | h, r] ]}}",
     "size": 22, "after": 6},
    {"text": "x = (j, a, w): marca, edad y desgaste; h: el coche con el que se "
             "queda; ε valor extremo → logit anidado.", "size": XS,
     "color": MUTED, "after": 0},
], margin=0.3)
y += eqh + 0.35

textbox(sl, x, y, COLW, 0.5, [label("Árbol de decisiones de un hogar con coche")])
y += 0.5
nh, step = 0.95, 1.55
cxs = [x + COLW * f for f in (1 / 6, 3 / 6, 5 / 6)]


def node(cx, yy, w, text, hidden=False, root=False):
    shp = rect(sl, cx - w / 2, yy, w, nh,
               fill=(CREAM if hidden else (NAVY if root else PALE)),
               line=(ORANGE if hidden else None), lw=1.5, r=0.06)
    paras = [{"text": t, "size": 19,
              "color": ORANGE if hidden else (WHITE if root else NAVY),
              "b": hidden or root, "align": PP_ALIGN.CENTER, "after": 0,
              "font": HEAD if (hidden or root) else BODY}
             for t in text.split("\n")]
    put(shp, paras, anchor=MSO_ANCHOR.MIDDLE, margin=0.05)
    return (cx, yy, yy + nh)


def link(parent, children):
    pcx, _, pb = parent
    mid = pb + (step - nh) / 2
    hline(sl, pcx, pb, pcx, mid, color=MUTED)
    xs = [c[0] for c in children]
    if len(xs) > 1:
        hline(sl, min(xs), mid, max(xs), mid, color=MUTED)
    for c in children:
        hline(sl, c[0], mid, c[0], c[1], color=MUTED)


root = node(x + COLW / 2, y, 5.0, "Hogar con coche (j, a, w)", root=True)
l1 = [node(cxs[0], y + step, 3.2, "Deshacerse"),
      node(cxs[1], y + step, 3.2, "Mantener"),
      node(cxs[2], y + step, 3.2, "Intercambiar")]
link(root, l1)
off, cw = 0.86, 1.66
d0 = [node(cxs[0] - off, y + 2 * step, cw, "Vender\nD = 0"),
      node(cxs[0] + off, y + 2 * step, cw, "Chatarrizar\nD = 1")]
link(l1[0], d0)
m0 = [node(cxs[1] - off, y + 2 * step, cw, "Reparar\nr = 1", True),
      node(cxs[1] + off, y + 2 * step, cw, "No reparar\nr = 0", True)]
link(l1[1], m0)
t1 = node(cxs[2], y + 2 * step, 3.2, "Vende o chatarriza\nel actual")
link(l1[2], [t1])
t2 = node(cxs[2], y + 3 * step, 3.2, "Compra otro (i, b, y)")
link(t1, [t2])
t3 = [node(cxs[2] - off, y + 4 * step, cw, "Reparar\nr = 1", True),
      node(cxs[2] + off, y + 4 * step, cw, "No reparar\nr = 0", True)]
link(t2, t3)
textbox(sl, x, y + 3 * step + 0.05, COLW * 0.58, 1.9, [
    {"text": "[[En naranja]]: lo que el econometrista no ve. Los coches nuevos "
             "y los de la última edad no se reparan.", "size": XS,
     "color": MUTED, "after": 0}])
y += 4 * step + nh + 0.4

dh = 3.75
put(rect(sl, x, y, COLW, dh, fill=CREAM, name="Desgaste y avería"), [
    label("Cómo entra la reparación", color=ORANGE),
    {"text": "Desgaste:   {{w′ = w + δ_j − }}<<κ·r>>{{ + η,     η ~ N(0, σ_η^{2})}}",
     "size": 22, "after": 4},
    {"text": "Avería:   {{s(j, a, w) = Λ(α_{0j} + α_{1j}·a + w)}}", "size": 22,
     "after": 8},
    {"text": "{{α_{1j}·a}} es envejecimiento que no se repara; {{w}} es desgaste "
             "que sí. Reparar baja {{w′}} en {{κ}}: menos averías, más utilidad "
             "y un mejor precio de reventa.", "size": S, "after": 8},
    {"text": "**Equilibrio:** en cada mercado {{m}}, los precios de los usados "
             "{{P_m(j, a, w)}} vacían el mercado y la distribución de coches es "
             "estacionaria.", "size": S, "after": 0},
], margin=0.3)
y += dh + 0.45
textbox(sl, x, y, COLW, 0.5, [label("Precios predichos de los usados")])
y += 0.5
edades = sorted(PRECIOS[("rep", 0)])
cdp = CategoryChartData()
cdp.categories = [str(a) for a in edades]
for nm, key in [("Light brown, con reparación", ("rep", 0)),
                ("Heavy brown, con reparación", ("rep", 1)),
                ("Light brown, Gillingham", ("gill", 0)),
                ("Heavy brown, Gillingham", ("gill", 1))]:
    cdp.add_series(nm, [round(PRECIOS[key][a], 1) for a in edades])
PCH = 5.6
gp = sl.shapes.add_chart(XL_CHART_TYPE.LINE, I(x), I(y), I(COLW), I(PCH), cdp)
gp.name = "Gráfica precios"
cp = gp.chart
cp.font.size = Pt(XS)
cp.font.name = BODY
cp.has_legend = True
cp.legend.position = XL_LEGEND_POSITION.TOP
cp.legend.include_in_layout = False
cp.legend.font.size = Pt(15)
for s_, c_, ds in zip(cp.plots[0].series, [BLUE, NAVY, BLUE, NAVY],
                      [None, None, MSO_LINE_DASH_STYLE.DASH,
                       MSO_LINE_DASH_STYLE.DASH]):
    s_.format.line.color.rgb = c_
    s_.format.line.width = Pt(4 if ds is None else 2.5)
    if ds:
        s_.format.line.dash_style = ds
    s_.smooth = False
    s_.marker.style = None
    from pptx.enum.chart import XL_MARKER_STYLE
    s_.marker.style = XL_MARKER_STYLE.NONE
vp, cpa = cp.value_axis, cp.category_axis
vp.has_major_gridlines = True
vp.major_gridlines.format.line.color.rgb = RGBColor(0xE2, 0xE7, 0xEE)
vp.format.line.fill.background()
vp.tick_labels.font.size = Pt(14)
vp.tick_labels.font.color.rgb = MUTED
vp.has_title = True
vp.axis_title.text_frame.text = "miles de DKK"
vr = vp.axis_title.text_frame.paragraphs[0].runs[0]
vr.font.size = Pt(14)
vr.font.bold = False
vr.font.color.rgb = MUTED
cpa.tick_labels.font.size = Pt(14)
cpa.tick_labels.font.color.rgb = MUTED
cpa.format.line.color.rgb = RULE
cpa.tick_label_spacing = 2
cpa.has_title = True
cpa.axis_title.text_frame.text = "edad del coche (años)"
cr = cpa.axis_title.text_frame.paragraphs[0].runs[0]
cr.font.size = Pt(14)
cr.font.bold = False
cr.font.color.rgb = MUTED
y += PCH + 0.05
textbox(sl, x, y, COLW, 1.0, [
    {"text": "Precio promedio por marca y edad en el equilibrio del mercado "
             "base, contra el modelo de Gillingham con los mismos parámetros. "
             "Con reparación, los coches viejos conservan más valor.",
     "size": XS, "color": MUTED, "after": 0}])
y += 1.0
COL_END.append(y)

# ================================================================ COLUMNA 3
x, y = COLX[2], Y0
y = section(sl, x, y, 5, "¿Se separan las campanas?")
textbox(sl, x, y, COLW, 1.7, [
    {"text": "No vemos si el coche se reparó, pero sí su desgaste un año "
             "después. Ese desgaste sale de una de dos campanas, [[reparó]] o "
             "**no reparó**, y lo que observamos es su **mezcla**.",
     "after": 0}])
y += 1.75

d_, k_, sg = 0.075, 0.25, 0.15
p_bar, p_car = 0.45, 0.15


def npdf(z, m, s):
    return math.exp(-0.5 * ((z - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


grid = [round(-0.75 + 0.01 * i, 3) for i in range(141)]
cd = XyChartData()
for nm, fn in [
    ("Reparó (no se ve)", lambda z: npdf(z, d_ - k_, sg)),
    ("No reparó (no se ve)", lambda z: npdf(z, d_, sg)),
    ("Se ve: reparar barato", lambda z: p_bar * npdf(z, d_ - k_, sg)
     + (1 - p_bar) * npdf(z, d_, sg)),
    ("Se ve: reparar caro", lambda z: p_car * npdf(z, d_ - k_, sg)
     + (1 - p_car) * npdf(z, d_, sg)),
]:
    sr = cd.add_series(nm)
    for z in grid:
        sr.add_data_point(z, fn(z))
gf = sl.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_SMOOTH_NO_MARKERS,
                         I(x), I(y), I(COLW), I(7.8), cd)
gf.name = "Gráfica mezcla"
ch = gf.chart
ch.font.size = Pt(XS)
ch.font.name = BODY
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.TOP
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(XS)
for s_, c_, w_, ds in zip(
        ch.plots[0].series, [ORANGE, BLUE, NAVY, MUTED], [2.5, 2.5, 5, 5],
        [MSO_LINE_DASH_STYLE.DASH, MSO_LINE_DASH_STYLE.DASH, None,
         MSO_LINE_DASH_STYLE.SQUARE_DOT]):
    s_.format.line.color.rgb = c_
    s_.format.line.width = Pt(w_)
    if ds:
        s_.format.line.dash_style = ds
    s_.smooth = True
va, ca = ch.value_axis, ch.category_axis
va.visible = False
va.has_major_gridlines = False
ca.minimum_scale, ca.maximum_scale = -0.75, 0.65
ca.tick_labels.font.size = Pt(XS)
ca.tick_labels.font.color.rgb = MUTED
ca.format.line.color.rgb = RULE
ca.has_title = True
ca.axis_title.text_frame.text = "desgaste un año después, w′  (hoy w = 0)"
rr = ca.axis_title.text_frame.paragraphs[0].runs[0]
rr.font.size = Pt(XS)
rr.font.bold = False
rr.font.color.rgb = MUTED
y += 7.85
textbox(sl, x, y, COLW, 2.6, [
    {"text": "**La variable excluida.** Comparamos mercados con distinto "
             "precio de reparar R_m. El precio cambia __cuántos__ reparan (el "
             "peso de cada campana), pero __no__ cómo reparar mueve el desgaste "
             "(su forma). Esa variación separa las campanas.", "size": S,
     "after": 4},
    {"text": "Ilustración con κ = 0.25 y σ_η = 0.15 del Monte Carlo; pesos de "
             "reparación ilustrativos.", "size": XS, "i": True,
     "color": MUTED, "after": 0}])
y += 2.75

y = section(sl, x, y, 6, "La verosimilitud")
textbox(sl, x, y, COLW, 1.0, [
    {"text": "Un hogar empieza en (j, a, w) y al año siguiente vemos qué hizo "
             "(e) y con qué coche quedó (δ′). Su contribución:", "size": S,
     "after": 0}])
y += 1.05
lh = 2.3
put(rect(sl, x, y, COLW, lh, fill=WHITE, line=RULE), [
    {"text": "{{ℓ = s·Π(δ′ | avería)·𝟙[D = 1]  +  (1 − s)·Σ_{w′} }}"
             "<<Z(w′ | j, a, w)>>{{·Π(e, δ′ | j, a+1, w′)}}",
     "size": 22, "after": 6},
    {"text": "Primer término: el coche se averió (solo si lo vemos "
             "chatarrizado). Segundo: sobrevivió; si sigue circulando vemos "
             "{{w′}} y la suma tiene un solo término.", "size": XS,
     "color": MUTED, "after": 0},
], margin=0.25, anchor=MSO_ANCHOR.MIDDLE)
y += lh + 0.25
zh = 2.8
put(rect(sl, x, y, COLW, zh, fill=CREAM, line=ORANGE, lw=1.75,
         name="Mezcla de reparación"), [
    label("La mezcla de reparación  (Hu & Xin, ec. 6.4)", color=ORANGE),
    {"text": "{{Z(w′ | j, a, w) = Σ_{r=0,1} }}<<Π(r | j, a, w; P_m, R_m)>>"
             "{{ · f(w′ | j, a, w, r)}}", "size": 22, "after": 6},
    {"text": "No vemos r: sumamos sus dos posibilidades. Los pesos (prob. de "
             "reparar) salen del modelo y cambian con el mercado; las "
             "densidades no.", "size": XS, "color": TEXT, "after": 0},
], margin=0.25, anchor=MSO_ANCHOR.MIDDLE)
y += zh + 0.35

y += 0.45
textbox(sl, x, y, COLW, 0.5, [label("Estimación: punto fijo anidado")])
y += 0.5
steps = [("1 · Proponer θ", "26 parámetros"),
         ("2 · Bellman", "qué hace cada hogar  (Rust)"),
         ("3 · Precios P_m(θ)", "equilibrio por mercado  (Gillingham)"),
         ("4 · LL(θ)", "con la mezcla  (Hu & Xin)")]
sw = (COLW - 3 * 0.32) / 4
for k, (t, d) in enumerate(steps):
    bx = x + k * (sw + 0.32)
    put(rect(sl, bx, y, sw, 1.45, fill=PALE), [
        {"text": t, "size": 17, "b": True, "color": BLUE, "font": HEAD,
         "after": 2},
        {"text": d, "size": 15, "color": TEXT, "after": 0},
    ], margin=0.12)
    if k < 3:
        textbox(sl, bx + sw, y + 0.4, 0.32, 0.6, [
            {"text": "→", "size": 20, "color": MUTED,
             "align": PP_ALIGN.CENTER, "after": 0}])
y += 1.5
textbox(sl, x, y, COLW, 0.5, [
    {"text": "↻ se repite hasta maximizar LL(θ); los precios de los usados "
             "nunca se observan, salen del equilibrio.", "size": XS,
     "color": MUTED, "after": 0}])
y += 0.5
COL_END.append(y)

# ================================================================ COLUMNA 4
x, y = COLX[3], Y0
y = section(sl, x, y, 7, "Diseños del Monte Carlo")
facts = [("1,000", ["réplicas"]),
         ("3", ["mercados: R·e^ζ,", "ζ ∈ {−0.3, 0, 0.3}"]),
         ("10,000", ["hogares × 4 años", "por mercado"])]
fw = COLW / 3
for k, (n, labs) in enumerate(facts):
    textbox(sl, x + k * fw, y, fw - 0.15, 1.6, [
        {"text": n, "size": 38, "color": ORANGE, "font": HEAD, "after": 0}] +
        [{"text": lb, "size": XS, "color": MUTED, "after": 0} for lb in labs])
y += 1.75
textbox(sl, x, y, COLW, 1.05, [
    {"text": "Simulamos un panel y guardamos **todo**. Cada diseño **tapa "
             "columnas** del mismo panel: las diferencias son solo de "
             "información.", "size": S, "after": 0}])
y += 1.1

cols = ["hogar", "tipo · mercado", "marca · edad", "w", "decisión", "w′",
        "motivo", "r"]
cwid = [0.95, 1.65, 1.5, 0.85, 2.18, 0.85, 1.4, 0.85]
scale = COLW / sum(cwid)
cwid = [c * scale for c in cwid]
textbox(sl, x, y, COLW, 0.45, [label("La base: una fila por hogar y año")])
y += 0.45
datos = [cols,
         ["17", "1 · barato", "LB · 5", "0.30", "mantiene", "0.21", "—",
          "1"],
         ["42", "2 · caro", "HB · 9", "0.85", "chatarriza y compra", "?",
          "avería", "0"]]
mini_table(sl, x, y, COLW, datos, cwid, 0.7, sizes=[XS] * 8,
           name="Base de datos", aligns=[PP_ALIGN.CENTER] * 8)
y += 0.7 * 3 + 0.4

textbox(sl, x, y, COLW, 0.45, [label("Qué ve cada estimador")])
y += 0.45
tapa = {"Diseño 1": set(), "Diseño 2": {"r"}, "Diseño 3": {"r", "motivo"},
        "Gillingham": {"w", "w′", "r"}}
mask = [["", *cols[1:]]]
for dname, hid in tapa.items():
    mask.append([dname] + ["" if c in hid else "✓" for c in cols[1:]])
mw = [cwid[0] + 0.85] + [(COLW - cwid[0] - 0.85) * c / sum(cwid[1:])
                         for c in cwid[1:]]


def estilo_mask(i, j, val, st):
    if i == 0:
        st["size"] = 14
        return
    if j == 0:
        st.update(b=True, font=HEAD, color=NAVY, size=17)
        if i == 4:
            st["color"] = ORANGE
        return
    if val == "":
        st["fill"] = REDACT
    else:
        st.update(color=BLUE, size=18)


mini_table(sl, x, y, COLW, mask, mw, 0.7, sizes=[17] * 8, name="Máscaras",
           aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 7, styler=estilo_mask)
y += 0.7 * 5 + 0.1
rect(sl, x, y + 0.08, 0.32, 0.22, fill=REDACT, r=0)
textbox(sl, x + 0.42, y, COLW - 0.42, 0.9, [
    {"text": "tapado. Gillingham: su modelo sin desgaste ni reparación. "
             "Nadie ve los precios de los usados.", "size": XS,
     "color": MUTED, "after": 0}])
y += 0.95

y = section(sl, x, y, 8, "Resultados preliminares")
textbox(sl, x, y - 0.15, COLW, 0.45, [
    {"text": NOTA_RESULTADOS, "size": XS, "i": True, "color": ORANGE,
     "after": 0}])
y += 0.4

textbox(sl, x, y, COLW, 1.35, [
    {"text": f"{100 * TASA_V:.1f}%  verdadera   vs   "
             f"{100 * TASA['diseno_3']:.1f}%  estimada", "size": 34,
     "color": ORANGE, "font": HEAD, "after": 0},
    {"text": "tasa de reparación, sin ver la reparación ni el motivo de salida "
             "(diseño 3)", "size": XS, "color": MUTED, "after": 0}])
y += 1.45

pars = [("κ  efecto de reparar", "kappa", 3),
        ("σ_η  choque al desgaste", "sigma_eta", 3),
        ("δ  desgaste anual", "delta_j0", 3),
        ("σ_r  escala de reparar", "sigma_rep", 2),
        ("u_w  utilidad del desgaste", "u_w", 2)]
rows = [["Parámetro", "Verdad", "D1", "D2", "D3"]]
for nm, p, nd in pars:
    rows.append([nm, f"{VERDAD[p]:.{nd}f}"] +
                [f"{EST[(d, p)]:.{nd}f}" for d in ("diseno_1", "diseno_2",
                                                     "diseno_3")])
tw = [COLW - 4 * 1.3] + [1.3] * 4
mini_table(sl, x, y, COLW, rows, tw, 0.58, sizes=[17] * 5, name="Estimados",
           aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 4)
y += 0.58 * len(rows) + 0.35

keys = [("mu_t0", "μ  sensibilidad al precio"),
        ("u0_t0_j0", "u₀  utilidad base"),
        ("u1_t0_j0", "u₁  pérdida por edad"),
        ("acc_age_j0", "α₁  avería por edad"),
        ("tc_buy_t0", "T  costo de comprar")]
cd2 = CategoryChartData()
cd2.categories = [lab for _, lab in keys]
cd2.add_series("Gillingham (sin reparación)",
               [round(sesgo(GEST[("gill_completa", k)], GVERDAD, k), 1)
                for k, _ in keys])
cd2.add_series("Diseño 3 (con reparación)",
               [round(sesgo(EST[("diseno_3", k)], VERDAD, k), 1)
                for k, _ in keys])
FOOT = H_IN - 1.25
ch_h = FOOT - 0.3 - y
gf2 = sl.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, I(x), I(y), I(COLW),
                          I(ch_h), cd2)
gf2.name = "Gráfica sesgo"
c2 = gf2.chart
c2.font.size = Pt(XS)
c2.font.name = BODY
c2.has_title = True
c2.chart_title.text_frame.text = "Sesgo, % respecto al valor verdadero"
tr = c2.chart_title.text_frame.paragraphs[0].runs[0]
tr.font.size = Pt(S)
tr.font.bold = True
tr.font.color.rgb = NAVY
tr.font.name = HEAD
c2.has_legend = True
c2.legend.position = XL_LEGEND_POSITION.BOTTOM
c2.legend.include_in_layout = False
c2.legend.font.size = Pt(XS)
pl = c2.plots[0]
pl.gap_width = 55
pl.overlap = -5
pl.has_data_labels = True
dl = pl.data_labels
dl.number_format = '+0"%";-0"%"'
dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(15)
for s_, c_ in zip(pl.series, [ORANGE, BLUE]):
    s_.format.fill.solid()
    s_.format.fill.fore_color.rgb = c_
    s_.invert_if_negative = False
va2, ca2 = c2.value_axis, c2.category_axis
va2.minimum_scale, va2.maximum_scale = -55, 25
va2.major_unit = 20
va2.has_major_gridlines = True
va2.major_gridlines.format.line.color.rgb = RGBColor(0xE2, 0xE7, 0xEE)
va2.tick_labels.font.size = Pt(14)
va2.tick_labels.font.color.rgb = MUTED
va2.tick_labels.number_format = '0"%"'
va2.tick_labels.number_format_is_linked = False
ca2.reverse_order = True
ca2.tick_labels.font.size = Pt(15)
ca2.format.line.color.rgb = RULE
ca2.tick_label_position = XL_TICK_LABEL_POSITION.LOW
y += ch_h
COL_END.append(y)

# ---- referencias (pie)
hline(sl, MARGIN, FOOT, W_IN - MARGIN, FOOT)
textbox(sl, MARGIN, FOOT + 0.1, W_IN - 2 * MARGIN, 1.0, [
    {"text": "Gillingham, K., Iskhakov, F., Munk-Nielsen, A., Rust, J. y "
             "Schjerning, B. (2022). Equilibrium trade in automobiles. "
             "__Journal of Political Economy__.   ·   Hu, Y. y Xin, Y. (2024). "
             "Identification and estimation of dynamic structural models with "
             "unobserved choices. __Journal of Econometrics__.   ·   Rust, J. "
             "(1987). Optimal replacement of GMC bus engines. "
             "__Econometrica__.", "size": 15, "color": MUTED, "after": 0}])

print("columnas terminan en:", " ".join(f"{v:.2f}" for v in COL_END),
      f"(límite {FOOT - 0.3:.2f}); alto gráfica sesgo {ch_h:.2f}")
prs.save(ARGS.out)
print("guardado:", ARGS.out)
