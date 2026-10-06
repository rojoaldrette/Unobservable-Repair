# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/analisis/tablas.py
# Goal:           Tablas tipo regresión (CSV, LaTeX y Markdown)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           06/10/2026
#
# _____________________________________________________________________________
'''
    tabla_parametros   filas = parámetros, columnas = verdad y cada estimador; celdas
                       "estimado***" con (se) abajo.  Estrellas: H0 parámetro = 0
                       (* 10%, ** 5%, *** 1%).  Pie: observaciones, log-verosimilitud,
                       convergencia.  Versión "principal" (los parámetros clave) y "completa".
    tabla_sesgo        lo mismo pero estimado − verdad y z = (estimado − verdad)/se: la
                       prueba relevante en una simulación (¿se recupera la verdad?).
    tabla_mercado      estadísticas de equilibrio verdaderas y en θ̂ (sin coche, reparación,
                       accidentes, edad media, precio medio, RMSE de P).
    tabla_mc           con varias réplicas: sesgo, RMSE, se medio, sd del MC, cobertura 95%.

Cada tabla se escribe como <nombre>.csv (números sin formato), <nombre>.tex (booktabs)
y <nombre>.md.
'''

import os

import numpy as np
import pandas as pd

from datos import etiqueta, mejores

NOMBRE = {
    "mu": "μ (utilidad del dinero)", "u0": "u0 (constante de utilidad)", "u1": "u1 (pendiente en edad)",
    "u_s": "u_s (desutilidad de s)", "tc_buy": "Tb (costo del comprador)",
    "tc_buy_nocar": "Tb extra desde sin coche", "tc_sell": "Ts (costo del vendedor)",
    "tc_sell_inspect": "Ts en año de inspección", "sigma_repair": "σ del choque de reparar",
    "s_const": "s: constante", "s_age": "s: edad", "s_persist": "s: persistencia",
    "s_repair": "s: efecto de reparar", "s_sigma": "s: sd del choque η",
    "sigma_sell": "σ vender/chatarrear", "acc_int": "accidente: constante", "acc_age": "accidente: edad",
}
TIPO = {"low_couple_poor": "pareja pobre", "low_couple_rich": "pareja rica",
        "low_single_poor": "soltero pobre", "low_single_rich": "soltero rico"}
MARCA = {"0": "LB", "1": "LG", "2": "HB"}
PRINCIPALES = ("mu", "tc_buy", "tc_buy_nocar", "tc_sell", "tc_sell_inspect", "sigma_repair",
               "u_s", "s_repair", "s_sigma", "sigma_sell")
ORDEN = ("verdad", "oraculo", "hx", "gill_parcial|modelo_fin", "gill_completa|modelo_fin",
         "gill_parcial", "gill_completa")


def _base(lab):
    for k in sorted(NOMBRE, key=len, reverse=True):
        if lab == k or lab.startswith(k + "_"):
            return k, lab[len(k) + 1:]
    return lab, ""


def nombre_param(lab, tipos):
    k, resto = _base(lab)
    partes = resto.split("_") if resto else []
    extra = []
    for p in partes:
        if p.startswith("t") and p[1:].isdigit():
            i = int(p[1:])
            extra.append(TIPO.get(tipos[i], tipos[i]) if i < len(tipos) else p)
        else:
            extra.append(MARCA.get(p, p))
    return NOMBRE.get(k, k) + (f" [{', '.join(extra)}]" if extra else "")


def _estrellas(z):
    if not np.isfinite(z):
        return ""
    z = abs(z)
    return "***" if z > 2.576 else "**" if z > 1.96 else "*" if z > 1.645 else ""


# Escritura ______________________________________________________________

def _tex_escape(s):
    return (str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("μ", r"$\mu$").replace("σ", r"$\sigma$").replace("η", r"$\eta$")
            .replace("θ̂", r"$\hat\theta$").replace("−", "$-$"))


def escribir(df, out, nombre, titulo, nota=""):
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{nombre}.csv"), index=False)
    cols = list(df.columns)
    # Markdown
    md = [f"**{titulo}**", "", "| " + " | ".join(map(str, cols)) + " |",
          "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    if nota:
        md += ["", nota]
    with open(os.path.join(out, f"{nombre}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    # LaTeX (booktabs)
    tex = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{_tex_escape(titulo)}}}",
           r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
           " & ".join(_tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for row in df.itertuples(index=False):
        tex.append(" & ".join("" if pd.isna(v) else _tex_escape(v) for v in row) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    if nota:
        tex.append(rf"\par\smallskip\footnotesize {_tex_escape(nota)}")
    tex.append(r"\end{table}")
    with open(os.path.join(out, f"{nombre}.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex) + "\n")


# Tablas ______________________________________________________________

def _juntar(corridas, rep):
    pars, res = [], []
    for c in corridas:
        if c.parametros is not None:
            d = mejores(c.parametros)
            d = d[d["rep"] == (rep if rep is not None else d["rep"].min())]
            pars.append(d)
        if c.resumen is not None:
            r = c.resumen
            res.append(r[r["rep"] == (rep if rep is not None else r["rep"].min())])
    return (pd.concat(pars) if pars else pd.DataFrame(),
            pd.concat(res) if res else pd.DataFrame())


def _orden_est(estimadores):
    return sorted(estimadores, key=lambda e: ORDEN.index(e) if e in ORDEN else len(ORDEN))


def tabla_parametros(corridas, out, tipos, rep=None, principales=True, sesgo=False):
    p, r = _juntar(corridas, rep)
    if p.empty:
        return
    if principales:
        p = p[p["parametro"].map(lambda l: _base(l)[0] in PRINCIPALES)]
    labs = list(dict.fromkeys(p["parametro"]))
    ests = _orden_est(p["estimador"].unique())
    filas = []
    for lab in labs:
        dl = p[p["parametro"] == lab]
        fila1 = {"Parámetro": nombre_param(lab, tipos), "Verdad": f"{dl['verdad'].iloc[0]:.4f}"}
        fila2 = {"Parámetro": "", "Verdad": ""}
        for e in ests:
            de = dl[dl["estimador"] == e]
            if de.empty:
                fila1[etiqueta(e)] = fila2[etiqueta(e)] = ""
                continue
            est, se, tv = (float(de[k].iloc[0]) for k in ("estimado", "se", "verdad"))
            if sesgo:
                zv = (est - tv) / se if se > 0 else np.nan
                fila1[etiqueta(e)] = f"{est - tv:+.4f}"
                fila2[etiqueta(e)] = f"[z = {zv:.2f}]" if np.isfinite(zv) else ""
            else:
                z = est / se if se > 0 else np.nan
                fila1[etiqueta(e)] = f"{est:.4f}{_estrellas(z)}"
                fila2[etiqueta(e)] = f"({se:.4f})" if np.isfinite(se) else "(n.d.)"
        filas += [fila1, fila2]
    tab = pd.DataFrame(filas)
    # Pie
    for nombre_fila, col, fmt in (("Observaciones", "n_obs", "{:,.0f}"), ("Log-verosimilitud", "ll", "{:,.1f}"),
                                  ("Convergió (BHHH)", "convergio", "{}")):
        fila = {"Parámetro": nombre_fila, "Verdad": ""}
        for e in ests:
            de = r[r["estimador"] == e] if not r.empty else r
            fila[etiqueta(e)] = fmt.format(de[col].iloc[0]) if not de.empty and col in de else ""
        tab = pd.concat([tab, pd.DataFrame([fila])], ignore_index=True)
    sufijo = ("sesgo" if sesgo else "parametros") + ("_principal" if principales else "_completa")
    titulo = ("Estimado − verdad" if sesgo else "Estimaciones de los parámetros estructurales") + \
             (" (principales)" if principales else "")
    nota = ("[z] = (estimado − verdad)/se." if sesgo else
            "Errores estándar (BHHH) entre paréntesis. * p<0.10, ** p<0.05, *** p<0.01 (H0: parámetro = 0).")
    escribir(tab, out, f"tabla_{sufijo}", titulo, nota)


def tabla_mercado(corridas, out, rep=None):
    _, r = _juntar(corridas, rep)
    if r.empty:
        return
    # Misma estadística con nombres distintos en cada modelo: una sola fila
    nombres = {"sin_coche": "Hogares sin coche", "tasa_reparacion": "Tasa de reparación (por coche)",
               "tasa_chatarreo": "Chatarreo endógeno (por coche)", "endo_rate": "Chatarreo endógeno (por coche)",
               "tasa_accidentes": "Tasa de accidentes (por coche)", "acc_rate": "Tasa de accidentes (por coche)",
               "share_endo": "Fracción voluntaria de las salidas", "edad_media": "Edad media del parque",
               "precio_medio": "Precio medio de usados (miles DKK)"}
    filas = {}
    for k, nom in nombres.items():
        fila = filas.setdefault(nom, {"Estadística": nom})
        for col_v in (f"{k}_verdad", f"{k}_verdad_gill"):
            if col_v in r and r[col_v].notna().any():
                fila["Verdad" if col_v.endswith("_verdad") else "Verdad (modelo de Gillingham)"] =                     f"{r[col_v].dropna().iloc[0]:.4f}"
        for e in _orden_est(r["estimador"].unique()):
            de = r[r["estimador"] == e]
            if f"{k}_est" in de and de[f"{k}_est"].notna().any():
                fila[etiqueta(e)] = f"{de[f'{k}_est'].iloc[0]:.4f}"
    filas = [f for f in filas.values() if len(f) > 1]
    fila = {"Estadística": "RMSE de P (miles DKK, ponderado por q)"}
    for e in _orden_est(r["estimador"].unique()):
        de = r[r["estimador"] == e]
        if "P_rmse" in de and de["P_rmse"].notna().any():
            fila[etiqueta(e)] = f"{de['P_rmse'].iloc[0]:.3f}"
    filas.append(fila)
    escribir(pd.DataFrame(filas), out, "tabla_mercado", "Equilibrio verdadero y en los parámetros estimados",
             "Promedio sobre regímenes de R (modelo_fin). Gillingham sobre datos con reparación no tiene RMSE de P "
             "comparable (otro espacio de estados).")


def tabla_mc(corridas, out):
    d = pd.concat([mejores(c.parametros) for c in corridas if c.parametros is not None])
    if d.empty or d["rep"].nunique() < 2:
        return
    filas = []
    for (e, lab), dd in d.groupby(["estimador", "parametro"], sort=False):
        tv = dd["verdad"].iloc[0]
        cov = ((dd["estimado"] - tv).abs() <= 1.96 * dd["se"]).where(dd["se"].notna())
        filas.append(dict(estimador=etiqueta(e), parametro=lab, verdad=tv, media=dd["estimado"].mean(),
                          sesgo=dd["estimado"].mean() - tv,
                          rmse=float(np.sqrt(((dd["estimado"] - tv) ** 2).mean())),
                          se_medio=dd["se"].mean(), sd_mc=dd["estimado"].std(),
                          cobertura_95=cov.mean(), replicas=dd["rep"].nunique()))
    tab = pd.DataFrame(filas)
    os.makedirs(out, exist_ok=True)
    tab.to_csv(os.path.join(out, "tabla_mc_numerica.csv"), index=False)
    fmt = tab.copy()
    for col in ("verdad", "media", "sesgo", "rmse", "se_medio", "sd_mc", "cobertura_95"):
        fmt[col] = fmt[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    escribir(fmt, out, "tabla_mc", "Monte Carlo: sesgo, RMSE y cobertura",
             "Cobertura: fracción de réplicas con |estimado − verdad| <= 1.96 se.")
