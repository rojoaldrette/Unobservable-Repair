# -*- coding: utf-8 -*-
"""Póster de tesis (A0 vertical) con python-pptx.

Uso:  python claude/poster/poster.py
Salida: claude/poster/poster_tesis.pptx

Los números de "resultados preliminares" vienen de la corrida prueba2
(claude/niu/final/output/modelo_tesis/montecarlo/prueba2, 2 réplicas).
Cuando estén las 1000 réplicas, actualizar el bloque DATOS.
"""
import math
import re
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "poster_tesis.pptx"

# ---------------------------------------------------------------- estilo
NAVY = RGBColor(0x1B, 0x27, 0x33)
ORANGE = RGBColor(0xD9, 0x5D, 0x14)
TEAL = RGBColor(0x1F, 0x6F, 0x80)
LIGHT = RGBColor(0xEE, 0xF2, 0xF5)
LIGHT_OR = RGBColor(0xFC, 0xEB, 0xDF)
TEXT = RGBColor(0x1B, 0x27, 0x33)
MUTED = RGBColor(0x5B, 0x67, 0x73)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SOFT = RGBColor(0xC9, 0xD6, 0xE0)
GRAY = RGBColor(0x9A, 0xA5, 0xAF)

HEAD = "Cambria"
BODY = "Calibri"
MATH = "Cambria Math"

B = 22       # cuerpo
S = 18       # pie / notas
H = 36       # encabezado de sección

W_IN, H_IN = 33.11, 46.81          # A0 vertical
MARGIN = 1.0
GAP = 0.8
COLW = (W_IN - 2 * MARGIN - 2 * GAP) / 3
COLX = [MARGIN + k * (COLW + GAP) for k in range(3)]

# ---------------------------------------------------------------- datos
# Promedio de las 2 réplicas de prueba2 (estimado/verdad - 1, en %).
VERDAD = {"mu": 0.1131, "u0": 3.649, "u1": -0.1459, "tb": 6.5944, "a1": 0.1804}
GILL = {"mu": (0.06718, 0.06468), "u0": (2.79883, 2.81519),
        "u1": (-0.10293, -0.10329), "tb": (7.25081, 7.73318),
        "a1": (0.14004, 0.13684)}
D3 = {"mu": (0.11703, 0.12144), "u0": (3.71322, 3.81144),
      "u1": (-0.15001, -0.15545), "tb": (6.43980, 6.48665),
      "a1": (0.18632, 0.18417)}
ETIQ = {"mu": "μ  sensibilidad al precio",
        "u0": "u₀  utilidad base",
        "u1": "u₁  pérdida por edad",
        "tb": "T_b  costo de comprar",
        "a1": "α₁  avería por edad"}


def sesgo(d, k):
    return 100 * ((d[k][0] + d[k][1]) / 2 / VERDAD[k] - 1)


# ---------------------------------------------------------------- helpers
def I(x):
    return Inches(x)


_MARK = re.compile(r"(\*\*.+?\*\*|\[\[.+?\]\]|__.+?__|\{\{.+?\}\})")


def runs_from(text):
    """**negrita**, [[naranja negrita]], __itálica__, {{fórmula}}."""
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
        else:
            out.append((piece, {}))
    return out


def set_bullet(p, char="•", color=ORANGE, indent=0.38):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Emu(I(indent)))))
    pPr.set("indent", str(-int(Emu(I(indent)))))
    buClr = etree.SubElement(pPr, qn("a:buClr"))
    clr = etree.SubElement(buClr, qn("a:srgbClr"))
    clr.set("val", str(color))
    buFont = etree.SubElement(pPr, qn("a:buFont"))
    buFont.set("typeface", "Arial")
    buChar = etree.SubElement(pPr, qn("a:buChar"))
    buChar.set("char", char)


def fill_tf(tf, paras, size=B, color=TEXT, font=BODY, align=PP_ALIGN.LEFT,
            after=8, line=None):
    """paras: lista de str o dict(text, size, color, bullet, align, after, font, b, i)."""
    first = True
    for item in paras:
        if isinstance(item, str):
            item = {"text": item}
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = item.get("align", align)
        p.space_after = Pt(item.get("after", after))
        if item.get("before"):
            p.space_before = Pt(item["before"])
        if line or item.get("line"):
            p.line_spacing = item.get("line", line)
        if item.get("bullet"):
            set_bullet(p, char=item.get("char", "•"),
                       color=item.get("bcolor", ORANGE))
        for txt, st in runs_from(item["text"]):
            r = p.add_run()
            r.text = txt
            f = r.font
            f.size = Pt(st.get("size", item.get("size", size)))
            f.name = st.get("font", item.get("font", font))
            f.bold = st.get("b", item.get("b", False))
            f.italic = st.get("i", item.get("i", False))
            f.color.rgb = st.get("color", item.get("color", color))


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


def box(slide, x, y, w, h, fill=LIGHT, radius=0.06, line=None, shape=None,
        name=None):
    shp = slide.shapes.add_shape(shape or MSO_SHAPE.ROUNDED_RECTANGLE,
                                 I(x), I(y), I(w), I(h))
    if shape is None:
        shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(2)
    shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def shape_text(shp, paras, anchor=MSO_ANCHOR.MIDDLE, margin=0.15, **kw):
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", I(margin))
    fill_tf(tf, paras, **kw)


def circle(slide, x, y, d, label, fill=ORANGE, size=30, color=WHITE):
    c = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    shape_text(c, [{"text": label, "b": True, "align": PP_ALIGN.CENTER,
                    "after": 0}], margin=0, size=size, color=color)
    return c


def section(slide, x, y, num, title, w=COLW, color=NAVY):
    circle(slide, x, y, 0.85, str(num), size=28)
    textbox(slide, x + 1.1, y - 0.07, w - 1.1, 1.0,
            [{"text": title, "b": True, "after": 0}],
            anchor=MSO_ANCHOR.MIDDLE, size=H, color=color, font=HEAD,
            name=f"Sección {num}")
    return y + 1.25


def arrow(slide, x, y, w, h, kind=MSO_SHAPE.RIGHT_ARROW, fill=GRAY):
    a = slide.shapes.add_shape(kind, I(x), I(y), I(w), I(h))
    a.fill.solid()
    a.fill.fore_color.rgb = fill
    a.line.fill.background()
    a.shadow.inherit = False
    return a


def style_chart_text(chart, size=S):
    chart.font.size = Pt(size)
    chart.font.name = BODY
    chart.font.color.rgb = TEXT


# ---------------------------------------------------------------- póster
prs = Presentation()
prs.slide_width = I(W_IN)
prs.slide_height = I(H_IN)
sl = prs.slides.add_slide(prs.slide_layouts[6])
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = WHITE

# ---- encabezado
box(sl, 0, 0, W_IN, 4.5, fill=NAVY, shape=MSO_SHAPE.RECTANGLE, name="Encabezado")
textbox(sl, MARGIN, 0.55, W_IN - 2 * MARGIN, 3.8, [
    {"text": "¿Qué tanto importa la reparación?", "size": 76, "b": True,
     "font": HEAD, "color": WHITE, "after": 10},
    {"text": "Identificación de decisiones de reparación __no observadas__ en el "
             "mercado de coches con un modelo dinámico de elección discreta",
     "size": 36, "font": HEAD, "color": SOFT, "after": 18},
    {"text": "Rodrigo Antonio Aldrette Salas   ·   El Colegio de México   ·   "
             "Noviembre 2026", "size": 28, "color": WHITE, "after": 0},
], name="Título")

# ================================================================ COLUMNA 1
x, y = COLX[0], 5.2
y = section(sl, x, y, 1, "¿Por qué importa reparar?")
textbox(sl, x, y, COLW, 1.9, [
    {"text": "Reparar decide **cuánto vive** un coche y **cuánto vale** como "
             "usado. En México, un hogar gasta en promedio **19,389 pesos** al "
             "año en mantenimiento.", "bullet": True, "after": 10},
    {"text": "Pero la reparación casi **nunca se observa** en los datos: buena "
             "parte ocurre en talleres informales.", "bullet": True, "after": 0},
], name="Motivación")
y += 2.0
q = box(sl, x, y, COLW, 2.25, fill=LIGHT_OR, name="Pregunta")
shape_text(q, [
    {"text": "LA PREGUNTA", "size": S, "b": True, "color": ORANGE, "after": 4},
    {"text": "Si no vemos cuándo los hogares reparan, ¿podemos __inferirlo__ de "
             "lo que sí vemos? ¿Y qué tan equivocados estamos si lo ignoramos?",
     "size": 25, "font": HEAD, "after": 0},
], anchor=MSO_ANCHOR.MIDDLE, margin=0.3)
y += 2.25 + 0.55

# ---- aportación (bloque oscuro, el más llamativo de la columna)
ap_h = 10.1
box(sl, x, y, COLW, ap_h, fill=NAVY, radius=0.04, name="Aportación")
circle(sl, x + 0.35, y + 0.35, 0.85, "2", size=28)
textbox(sl, x + 1.45, y + 0.28, COLW - 1.8, 1.0,
        [{"text": "Mi aportación", "b": True, "after": 0}],
        anchor=MSO_ANCHOR.MIDDLE, size=H, color=WHITE, font=HEAD)
items = [
    ("Teoría",
     "La reparación como **decisión explícita** del hogar dentro de un "
     "equilibrio de coches nuevos y usados: el precio de reparar mueve el "
     "desgaste del coche y, por ahí, los precios de los usados."),
    ("Econometría",
     "Una **verosimilitud** que estima el modelo **sin ver la reparación** "
     "(ni si un coche salió por avería o por decisión), llevando la estrategia "
     "de Hu & Xin a un modelo de equilibrio."),
    ("Monte Carlo",
     "Mide **cuánto se pierde** por no observar la reparación y el **sesgo** "
     "de usar un modelo sin reparación (Gillingham et al.) cuando los hogares "
     "sí reparan."),
]
yy = y + 1.5
for k, (t, d) in enumerate(items):
    circle(sl, x + 0.4, yy + 0.05, 0.58, "abc"[k], fill=TEAL, size=S)
    textbox(sl, x + 1.25, yy, COLW - 1.6, 2.8, [
        {"text": t, "b": True, "color": ORANGE, "size": 26, "after": 2},
        {"text": d, "color": WHITE, "size": 23, "after": 0},
    ])
    yy += 2.85
y += ap_h + 0.55

# ---- literatura
y = section(sl, x, y, 3, "Sobre los hombros de…")
lit = [
    ("Rust (1987)",
     "Reemplazo de motores de autobús; **punto fijo anidado**: resolver "
     "Bellman dentro de la verosimilitud.",
     "La decisión sí se observaba."),
    ("Gillingham et al. (2022)",
     "Equilibrio de coches nuevos y usados en Dinamarca; los **precios de los "
     "usados** salen del modelo.",
     "Sin reparación: queda escondida en choques y costos."),
    ("Hu & Xin (2024)",
     "**Identificar decisiones no observadas** con un estado continuo y una "
     "variable excluida.",
     "La llevamos a un mercado en equilibrio."),
]
for name_, desc, gap in lit:
    c = box(sl, x, y, COLW, 2.15, fill=LIGHT)
    shape_text(c, [
        {"text": name_, "b": True, "color": TEAL, "size": 24, "font": HEAD,
         "after": 2},
        {"text": desc, "after": 3, "size": 20},
        {"text": "→ " + gap, "i": True, "color": ORANGE, "size": 20, "after": 0},
    ], anchor=MSO_ANCHOR.MIDDLE, margin=0.25)
    y += 2.15 + 0.25
COL1_END = y - 0.25

# ================================================================ COLUMNA 2
x, y = COLX[1], 5.2
y = section(sl, x, y, 4, "El modelo: un año de un hogar")
steps = [
    ("Inicio", "El hogar tiene un coche (marca j, edad a, desgaste w) o "
               "ninguno.", "observado"),
    ("Mercado", "Mantiene, vende, chatarriza o compra otro coche, nuevo o "
                "usado.", "observado"),
    ("Reparación", "Si se queda con un usado, decide reparar (r = 1) y paga el "
                   "precio R.", "NO observado"),
    ("Uso y avería", "Disfruta el coche; se descompone con probabilidad s, que "
                     "crece con la edad y el desgaste.", "solo a medias"),
    ("Desgaste", "Si sobrevive, cumple un año y su desgaste cambia: reparar lo "
                 "baja.", "observado"),
]
step_h = 1.3
line_x = x + 0.47
box(sl, line_x - 0.04, y + 0.4, 0.08, step_h * (len(steps) - 1),
    fill=SOFT, shape=MSO_SHAPE.RECTANGLE)
for k, (t, d, tag) in enumerate(steps):
    yy = y + k * step_h
    hidden = tag == "NO observado"
    circle(sl, x + 0.08, yy, 0.78, str(k + 1),
           fill=ORANGE if hidden else TEAL, size=24)
    textbox(sl, x + 1.25, yy - 0.05, COLW - 1.25, step_h, [
        {"text": t, "b": True, "size": 24,
         "color": ORANGE if hidden else NAVY, "after": 0},
        {"text": d, "size": 20, "after": 0},
    ])
    tagc = ORANGE if hidden else (MUTED if tag == "solo a medias" else TEAL)
    textbox(sl, x + COLW - 3.2, yy - 0.02, 3.2, 0.5,
            [{"text": tag.upper(), "b": True, "size": 16, "color": tagc,
              "align": PP_ALIGN.RIGHT, "after": 0}])
y += step_h * len(steps) + 0.25

eq = box(sl, x, y, COLW, 3.5, fill=LIGHT)
shape_text(eq, [
    {"text": "Las tres piezas clave", "b": True, "color": TEAL, "size": 22,
     "after": 4},
    {"text": "Avería:   {{s = Λ(α₀ + α₁·a + w)}}", "size": 22, "after": 0},
    {"text": "envejecer no se repara; el desgaste w sí", "size": S,
     "color": MUTED, "i": True, "after": 6},
    {"text": "Desgaste:   {{w′ = w + δ − κ·r + η}}", "size": 22, "after": 0},
    {"text": "reparar (r = 1) baja el desgaste en κ; η es suerte", "size": S,
     "color": MUTED, "i": True, "after": 6},
    {"text": "Decisión:   reparar si  {{μ·R  <  valor de un coche menos "
             "gastado}}", "size": 22, "after": 0},
], anchor=MSO_ANCHOR.MIDDLE, margin=0.3)
y += 3.5 + 0.2
textbox(sl, x, y, COLW, 1.4, [
    {"text": "**Equilibrio:** los precios de los usados P(j, a, w) se ajustan "
             "hasta que oferta = demanda en cada tipo de coche, como en "
             "Gillingham et al. Un coche menos desgastado vale más.",
     "after": 0},
])
y += 1.4 + 0.45

y = section(sl, x, y, 5, "La idea para ver lo invisible")
textbox(sl, x, y, COLW, 1.5, [
    {"text": "No vemos si el coche se reparó, pero sí su desgaste un año "
             "después. Ese desgaste viene de una de **dos campanas**: "
             "[[reparó]] o **no reparó**. Lo que observamos es su **mezcla**.",
     "after": 0},
])
y += 1.55

# gráfica de la mezcla (ilustrativa con κ, δ, σ_η verdaderos)
d, k_, sg = 0.075, 0.25, 0.15
p_bar, p_car = 0.45, 0.15


def npdf(z, m, s):
    return math.exp(-0.5 * ((z - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


grid = [round(-0.75 + 0.01 * i, 3) for i in range(141)]
cd = XyChartData()
series = [
    ("Reparó (no se ve)", lambda z: npdf(z, d - k_, sg)),
    ("No reparó (no se ve)", lambda z: npdf(z, d, sg)),
    ("Se ve: reparar barato", lambda z: p_bar * npdf(z, d - k_, sg)
     + (1 - p_bar) * npdf(z, d, sg)),
    ("Se ve: reparar caro", lambda z: p_car * npdf(z, d - k_, sg)
     + (1 - p_car) * npdf(z, d, sg)),
]
for nm, fn in series:
    sr = cd.add_series(nm)
    for z in grid:
        sr.add_data_point(z, fn(z))
gf = sl.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_SMOOTH_NO_MARKERS,
                         I(x), I(y), I(COLW), I(5.4), cd)
gf.name = "Gráfica mezcla"
ch = gf.chart
style_chart_text(ch)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.TOP
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(18)
cols = [ORANGE, TEAL, NAVY, GRAY]
widths = [2.5, 2.5, 5.5, 5.5]
dashes = [MSO_LINE_DASH_STYLE.DASH, MSO_LINE_DASH_STYLE.DASH, None,
          MSO_LINE_DASH_STYLE.SQUARE_DOT]
for s_, c_, w_, ds in zip(ch.plots[0].series, cols, widths, dashes):
    s_.format.line.color.rgb = c_
    s_.format.line.width = Pt(w_)
    if ds:
        s_.format.line.dash_style = ds
    s_.smooth = True
va, ca = ch.value_axis, ch.category_axis
va.visible = False
va.has_major_gridlines = False
ca.minimum_scale, ca.maximum_scale = -0.75, 0.65
ca.has_major_gridlines = False
ca.tick_labels.font.size = Pt(18)
ca.tick_labels.font.color.rgb = MUTED
ca.has_title = True
ca.axis_title.text_frame.text = "desgaste un año después, w′  (hoy w = 0)"
ca.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(18)
ca.axis_title.text_frame.paragraphs[0].runs[0].font.bold = False
ca.axis_title.text_frame.paragraphs[0].runs[0].font.color.rgb = MUTED
y += 5.45
idea = box(sl, x, y, COLW, 2.75, fill=LIGHT_OR)
shape_text(idea, [
    {"text": "La clave: una variable excluida", "b": True, "color": ORANGE,
     "size": 24, "after": 4},
    {"text": "Comparamos mercados con distinto **precio de reparar**. El precio "
             "cambia __cuántos__ reparan (el peso de cada campana) pero __no__ "
             "cómo reparar mueve el desgaste (la forma de las campanas). Esa "
             "variación separa las campanas y nos dice quién reparó.",
     "size": 20, "after": 0},
], anchor=MSO_ANCHOR.MIDDLE, margin=0.3)
y += 2.75
textbox(sl, x, y + 0.08, COLW, 0.5, [
    {"text": "Ilustración con los parámetros del Monte Carlo (κ = 0.25, "
             "σ_η = 0.15); pesos de reparación ilustrativos.", "size": 16,
     "color": MUTED, "i": True, "after": 0}])
COL2_END = y + 0.5

# ================================================================ COLUMNA 3
x, y = COLX[2], 5.2
y = section(sl, x, y, 6, "La verosimilitud, paso a paso")
textbox(sl, x, y, COLW, 1.5, [
    {"text": "Seguimos a cada hogar de un año al siguiente y preguntamos: "
             "__¿qué tan probable es, según el modelo, lo que vimos que "
             "pasó?__ Cada caso aporta un término distinto.", "after": 0},
])
y += 1.6
cases = [
    ("Mantuvo el coche",
     "Lo vemos un año más viejo, con su nuevo desgaste w′.",
     "{{ℓ = (1 − s) · Z(w′) · Π(mantener)}}"),
    ("Lo vendió",
     "Vemos el desgaste w′ con el que lo vendió.",
     "{{ℓ = (1 − s) · Z(w′) · Π(vender)}}"),
    ("Lo chatarrizó",
     "¿Avería o decisión? No lo sabemos, ni su w′: sumamos ambas historias.",
     "{{ℓ = s · Π(avería) + (1 − s) · Σ Z(w′) · Π(chatarrizar)}}"),
]
for t, d_, f in cases:
    c = box(sl, x, y, COLW, 2.1, fill=LIGHT)
    shape_text(c, [
        {"text": t, "b": True, "color": TEAL, "size": 26, "after": 2},
        {"text": d_, "size": 22, "after": 6},
        {"text": f, "size": 22, "after": 0},
    ], anchor=MSO_ANCHOR.MIDDLE, margin=0.3)
    y += 2.1 + 0.25
z = box(sl, x, y, COLW, 2.9, fill=NAVY)
shape_text(z, [
    {"text": "Aquí entra la reparación que no vemos", "b": True,
     "color": ORANGE, "size": 26, "after": 6},
    {"text": "{{Z(w′) = Π(r=1) · f(w′ | reparó)  +  Π(r=0) · f(w′ | no reparó)}}",
     "color": WHITE, "size": 22, "after": 8},
    {"text": "La probabilidad de reparar viene del modelo dinámico; la "
             "reparación se “promedia” en lugar de observarse.",
     "color": SOFT, "size": 22, "after": 0},
], anchor=MSO_ANCHOR.MIDDLE, margin=0.35)
y += 2.9 + 0.45

# ciclo del punto fijo anidado
textbox(sl, x, y, COLW, 0.6, [
    {"text": "Estimación: punto fijo anidado", "b": True, "size": 26,
     "color": NAVY, "font": HEAD, "after": 0}])
y += 0.75
bw, bh, gx, gy = (COLW - 0.9) / 2, 2.0, 0.9, 0.75
cells = [
    (0, 0, "1  Proponer θ", "preferencias, costos, avería, desgaste"),
    (1, 0, "2  Resolver Bellman", "qué haría cada hogar  (Rust)"),
    (1, 1, "3  Precios P(θ)", "que vacían el mercado de usados  (Gillingham)"),
    (0, 1, "4  Evaluar LL(θ)", "con la mezcla de reparación  (Hu & Xin)"),
]
for cx, cy, t, d_ in cells:
    bx, by = x + cx * (bw + gx), y + cy * (bh + gy)
    c = box(sl, bx, by, bw, bh, fill=WHITE, line=TEAL)
    shape_text(c, [
        {"text": t, "b": True, "color": TEAL, "size": 24, "after": 4},
        {"text": d_, "size": 19, "after": 0},
    ], anchor=MSO_ANCHOR.MIDDLE, margin=0.2)
arrow(sl, x + bw + 0.15, y + bh / 2 - 0.25, gx - 0.3, 0.5)
arrow(sl, x + bw + gx + bw / 2 - 0.25, y + bh + 0.12, 0.5, gy - 0.24,
      kind=MSO_SHAPE.DOWN_ARROW)
arrow(sl, x + bw + 0.15, y + bh + gy + bh / 2 - 0.25, gx - 0.3, 0.5,
      kind=MSO_SHAPE.LEFT_ARROW)
arrow(sl, x + bw / 2 - 0.25, y + bh + 0.12, 0.5, gy - 0.24,
      kind=MSO_SHAPE.UP_ARROW, fill=ORANGE)
y += 2 * bh + gy + 0.3
textbox(sl, x, y, COLW, 2.0, [
    {"text": "Se repite hasta maximizar  {{LL(θ) = Σ_celdas N_c · log ℓ_c}}. "
             "Agrupar hogares idénticos en celdas es exacto y hace la "
             "estimación mucho más barata. Los precios de los usados nunca se "
             "observan: salen del equilibrio.", "after": 0},
])
y += 2.0 + 0.3
dat = box(sl, x, y, COLW, 3.6, fill=LIGHT_OR, name="Datos")
shape_text(dat, [
    {"text": "¿Qué datos pide el método?", "b": True, "color": ORANGE,
     "size": 24, "after": 4},
    {"text": "Un panel de hogares con marca, edad y un **indicador del desgaste** "
             "de su coche (p. ej. inspecciones), en **mercados con distinto "
             "precio de reparar**. No hace falta ver los precios de los usados "
             "ni quién reparó.", "size": 21, "after": 0},
], anchor=MSO_ANCHOR.MIDDLE, margin=0.3)
y += 3.6
COL3_END = y

# ================================================================ BANDA INFERIOR
top = max(COL1_END, COL2_END, COL3_END) + 0.45
x = MARGIN
half = (W_IN - 2 * MARGIN - GAP) / 2

# ---- Monte Carlo
y = section(sl, x, top, 7, "Diseño del Monte Carlo", w=half)
stats = [("1,000", "réplicas"), ("3", "mercados con distinto precio de "
                                     "reparar"),
         ("10,000", "hogares × 4 años por mercado"), ("26", "parámetros")]
sw = (half - 3 * 0.25) / 4
for k, (n, lab) in enumerate(stats):
    c = box(sl, x + k * (sw + 0.25), y, sw, 1.9, fill=LIGHT)
    shape_text(c, [
        {"text": n, "b": True, "size": 42, "color": ORANGE, "font": HEAD,
         "align": PP_ALIGN.CENTER, "after": 0},
        {"text": lab, "size": 18, "color": MUTED, "align": PP_ALIGN.CENTER,
         "after": 0},
    ], margin=0.12)
y += 2.1
textbox(sl, x, y, half, 1.1, [
    {"text": "Simulamos la economía con reparación y guardamos **todo**. Cada "
             "diseño **borra columnas** del mismo panel: las diferencias entre "
             "diseños son solo de información, no de muestra.", "after": 0}])
y += 1.2

rows = [
    ("Estimador", "¿Ve la reparación?", "¿Ve por qué salió el coche?",
     "Pregunta que responde"),
    ("Diseño 1", "✓", "✓", "El mejor caso posible"),
    ("Diseño 2", "✗", "✓", "¿Cuánto cuesta no ver la reparación?"),
    ("Diseño 3", "✗", "✗", "Como en datos reales (registro danés)"),
    ("Gillingham", "sin reparación", "✓",
     "¿Cuánto sesgo da ignorar que se repara?"),
]
tbl_h = 4.4
gt = sl.shapes.add_table(len(rows), 4, I(x), I(y), I(half), I(tbl_h))
gt.name = "Tabla diseños"
tbl = gt.table
cw = [3.2, 2.6, 2.9, half - 8.7]
for j, w_ in enumerate(cw):
    tbl.columns[j].width = I(w_)
for i, row in enumerate(rows):
    tbl.rows[i].height = I(tbl_h / len(rows))
    for j, val in enumerate(row):
        cell = tbl.cell(i, j)
        cell.fill.solid()
        if i == 0:
            cell.fill.fore_color.rgb = NAVY
        elif i == 4:
            cell.fill.fore_color.rgb = LIGHT_OR
        else:
            cell.fill.fore_color.rgb = LIGHT if i % 2 else WHITE
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = cell.margin_right = I(0.15)
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if j in (1, 2) else PP_ALIGN.LEFT
        r = p.add_run()
        r.text = val
        f = r.font
        f.name = BODY
        f.size = Pt(19 if i == 0 else (20 if j == 3 or val == "sin reparación"
                                       else 22))
        f.bold = i == 0 or j == 0
        if i == 0:
            f.color.rgb = WHITE
        elif val == "✓":
            f.color.rgb = TEAL
            f.size = Pt(30)
        elif val == "✗":
            f.color.rgb = ORANGE
            f.size = Pt(30)
        else:
            f.color.rgb = TEXT
y += tbl_h + 0.1
textbox(sl, x, y, half, 0.8, [
    {"text": "Todos ven tipo de hogar, mercado, marca, edad, desgaste y compras "
             "y ventas. Nadie ve los precios de los usados.", "size": S,
     "color": MUTED, "i": True, "after": 0}])
MC_END = y + 0.8

# ---- Resultados preliminares
x = MARGIN + half + GAP
y = section(sl, x, top, 8, "Resultados preliminares", w=half)
textbox(sl, x, y - 0.2, half, 0.5, [
    {"text": "Prueba con 2 réplicas · las 1,000 réplicas están corriendo",
     "size": S, "i": True, "color": ORANGE, "after": 0}])
y += 0.4

cw2 = (half - 0.4) / 2
# A) inferencia de la reparación
c = box(sl, x, y, cw2, 3.9, fill=LIGHT)
shape_text(c, [
    {"text": "¿Adivinamos quién reparó?", "b": True, "color": TEAL,
     "size": 22, "after": 4},
    {"text": "0.75  →  0.91", "b": True, "size": 46, "color": ORANGE,
     "font": HEAD, "after": 2},
    {"text": "AUC antes y después de ver el desgaste del año siguiente "
             "(1 = acierto perfecto, 0.5 = moneda al aire)", "size": 18,
     "color": MUTED, "after": 10},
    {"text": "Igual con los parámetros estimados que con los verdaderos.",
     "size": 20, "after": 0},
], anchor=MSO_ANCHOR.TOP, margin=0.3)
# B) tasa de reparación
c = box(sl, x + cw2 + 0.4, y, cw2, 3.9, fill=LIGHT)
shape_text(c, [
    {"text": "¿Cuánto se repara?", "b": True, "color": TEAL, "size": 22,
     "after": 4},
    {"text": "27.1%  vs  27.0%", "b": True, "size": 46, "color": ORANGE,
     "font": HEAD, "after": 2},
    {"text": "tasa verdadera vs estimada sin ver la reparación ni el motivo "
             "de salida (diseño 3)", "size": 18, "color": MUTED, "after": 10},
    {"text": "κ estimado: 0.253 (verdadero 0.25).", "size": 20, "after": 0},
], anchor=MSO_ANCHOR.TOP, margin=0.3)
y += 3.9 + 0.25

# C) sesgo de ignorar la reparación
keys = ["mu", "u0", "u1", "a1", "tb"]
cd2 = CategoryChartData()
cd2.categories = [ETIQ[k] for k in keys]
cd2.add_series("Gillingham (sin reparación)", [round(sesgo(GILL, k), 1)
                                               for k in keys])
cd2.add_series("Diseño 3 (con reparación)", [round(sesgo(D3, k), 1)
                                             for k in keys])
gf2 = sl.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, I(x), I(y), I(half),
                          I(MC_END - y), cd2)
gf2.name = "Gráfica sesgo"
ch2 = gf2.chart
style_chart_text(ch2, 18)
ch2.has_title = True
ch2.chart_title.text_frame.text = ("Sesgo de ignorar la reparación "
                                   "(% respecto al valor verdadero)")
tr = ch2.chart_title.text_frame.paragraphs[0].runs[0]
tr.font.size = Pt(22)
tr.font.bold = True
tr.font.color.rgb = NAVY
ch2.has_legend = True
ch2.legend.position = XL_LEGEND_POSITION.BOTTOM
ch2.legend.include_in_layout = False
ch2.legend.font.size = Pt(18)
pl = ch2.plots[0]
pl.gap_width = 60
pl.overlap = -10
pl.has_data_labels = True
dl = pl.data_labels
dl.number_format = '+0"%";-0"%"'
dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(18)
dl.font.color.rgb = TEXT
for s_, c_ in zip(pl.series, [ORANGE, TEAL]):
    s_.format.fill.solid()
    s_.format.fill.fore_color.rgb = c_
    s_.invert_if_negative = False
va2, ca2 = ch2.value_axis, ch2.category_axis
va2.minimum_scale, va2.maximum_scale = -55, 25
va2.has_major_gridlines = True
va2.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xE3, 0xE8)
va2.tick_labels.font.size = Pt(16)
va2.tick_labels.font.color.rgb = MUTED
va2.tick_labels.number_format = '0"%"'
va2.tick_labels.number_format_is_linked = False
ca2.reverse_order = True
ca2.tick_labels.font.size = Pt(18)
ca2.format.line.color.rgb = GRAY
from pptx.enum.chart import XL_TICK_LABEL_POSITION  # noqa: E402
ca2.tick_label_position = XL_TICK_LABEL_POSITION.LOW
BAND_END = max(MC_END, y)

# ---- conclusiones y referencias
y = BAND_END + 0.5
foot_h = max(H_IN - 0.6 - y, 1.0)
c = box(sl, MARGIN, y, W_IN - 2 * MARGIN, foot_h, fill=LIGHT,
        name="Conclusiones")
concl_w = (W_IN - 2 * MARGIN) * 0.64
textbox(sl, MARGIN + 0.4, y + 0.3, concl_w - 0.4, foot_h - 0.6, [
    {"text": "Mensajes para llevar", "b": True, "size": 30, "color": NAVY,
     "font": HEAD, "after": 8},
    {"text": "La reparación **se puede recuperar** aunque no se observe: basta "
             "ver el desgaste y tener mercados con distinto precio de reparar.",
     "bullet": True, "size": 24, "after": 8},
    {"text": "**Ignorarla no es inocuo:** un modelo sin reparación sesga la "
             "sensibilidad al precio y las preferencias, que son la base de "
             "cualquier contrafactual de política.", "bullet": True,
     "size": 24, "after": 8},
    {"text": "**Siguiente paso:** 1,000 réplicas para medir sesgo, varianza y "
             "errores estándar de cada diseño.", "bullet": True, "size": 24,
     "after": 0},
], anchor=MSO_ANCHOR.MIDDLE)
textbox(sl, MARGIN + concl_w + 0.4, y + 0.3,
        W_IN - 2 * MARGIN - concl_w - 0.8, foot_h - 0.6, [
    {"text": "Referencias", "b": True, "size": 22, "color": NAVY, "after": 6},
    {"text": "Gillingham, K., Iskhakov, F., Munk-Nielsen, A., Rust, J. y "
             "Schjerning, B. (2022). Equilibrium trade in automobiles. "
             "__Journal of Political Economy__.", "size": 18, "after": 6},
    {"text": "Hu, Y. y Xin, Y. (2024). Identification and estimation of "
             "dynamic structural models with unobserved choices. __Journal of "
             "Econometrics__.", "size": 18, "after": 6},
    {"text": "Rust, J. (1987). Optimal replacement of GMC bus engines. "
             "__Econometrica__.", "size": 18, "after": 0},
], anchor=MSO_ANCHOR.MIDDLE)

print(f"columnas terminan en: {COL1_END:.2f} {COL2_END:.2f} {COL3_END:.2f}; "
      f"banda {top:.2f}-{BAND_END:.2f}; pie {BAND_END + 0.5:.2f}, alto {foot_h:.2f}")
prs.save(SALIDA)
print("guardado:", SALIDA)
