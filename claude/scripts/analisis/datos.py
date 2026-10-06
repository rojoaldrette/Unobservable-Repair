# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/analisis/datos.py
# Goal:           Leer las salidas de las estimaciones (CSV) y armar tablas comunes
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           06/10/2026
#
# _____________________________________________________________________________
'''
Una "corrida" es una carpeta de claude/output/estimaciones/<modelo>/<tag>/ con los CSV que
escriben modelo_fin/estimar.py y gillingham/estimar.py (ver docs/reporte_estimacion.md).

`leer_corrida(path)` -> Corrida con config, parametros, resumen, precios, distribucion,
ccps (DataFrames; None si el archivo no existe) y `modelo` ("modelo_fin" o "gillingham").

Nombres de los estimadores en las salidas:
    verdad          el equilibrio verdadero (no es un estimador)
    oraculo         modelo_fin, D0: r observada
    hx              modelo_fin, D1: r no observada (Hu & Xin dentro del DNFXP)
    gill_parcial    Gillingham, verosimilitud del paper (sin precios ni accidentes)
    gill_completa   Gillingham, oráculo (ve accidentes)
A los de Gillingham se les agrega el origen de los datos: gill_parcial|modelo_fin si se
estimó sobre el panel de modelo_fin (el ejercicio del sesgo).
'''

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ARCHIVOS = ("parametros", "resumen", "precios", "distribucion", "ccps")

ETIQUETA = {
    "verdad": "Verdad",
    "verdad_gill": "Verdad del modelo de Gillingham",
    "oraculo": "D0: r observada",
    "hx": "D1: r no observada",
    "gill_parcial": "Gillingham (paper)",
    "gill_completa": "Gillingham (ve accidentes)",
    "gill_parcial|modelo_fin": "Gillingham sobre datos con reparación",
    "gill_completa|modelo_fin": "Gillingham (ve accidentes) sobre datos con reparación",
}

MARCAS = {0: "Ligero gasolina (LB)", 1: "Ligero diésel (LG)", 2: "Pesado gasolina (HB)"}


@dataclass
class Corrida:
    path: str
    modelo: str
    config: dict
    parametros: pd.DataFrame = None
    resumen: pd.DataFrame = None
    precios: pd.DataFrame = None
    distribucion: pd.DataFrame = None
    ccps: pd.DataFrame = None

    @property
    def f(self):
        return tuple(self.config["tipos"]["f"])

    @property
    def nombre(self):
        return os.path.basename(os.path.normpath(self.path))


def leer_corrida(path):
    with open(os.path.join(path, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    modelo = "gillingham" if "datos" in config else "modelo_fin"
    c = Corrida(path=path, modelo=modelo, config=config)
    for nombre in ARCHIVOS:
        # parametros y resumen vienen en un archivo por bloque de réplicas: se juntan
        files = sorted(glob.glob(os.path.join(path, f"{nombre}.csv")) +
                       glob.glob(os.path.join(path, f"{nombre}_reps*.csv")))
        if files:
            df = pd.concat(map(pd.read_csv, files), ignore_index=True)
            if "rep" in df and "estimador" in df:
                keys = [k for k in ("rep", "estimador", "arranque", "parametro") if k in df]
                df = df.drop_duplicates(keys, keep="last")
            setattr(c, nombre, df)
    if modelo == "gillingham":
        # La verdad de Gillingham no es la de modelo_fin: "verdad_gill".  Si se estimó sobre
        # el panel de modelo_fin, los estimadores llevan "|modelo_fin".
        cruce = config.get("datos") == "modelo_fin"
        for nombre in ARCHIVOS:
            df = getattr(c, nombre)
            if df is None:
                continue
            col = "fuente" if "fuente" in df else "estimador"
            if col in df:
                v = df[col].astype(str)
                df[col] = np.where(v == "verdad", "verdad_gill", v + "|modelo_fin" if cruce else v)
    return c


def etiqueta(fuente):
    return ETIQUETA.get(fuente, fuente)


# Agregados ______________________________________________________________

def agregar_tipos(df, f, valor="q", claves=("fuente", "estado", "j", "a")):
    # Suma sobre tipos ponderando por f (para q).  df con columna `tipo`.
    w = df["tipo"].map(dict(enumerate(f)))
    out = df.assign(_v=df[valor] * w).groupby(list(claves), as_index=False)["_v"].sum()
    return out.rename(columns={"_v": valor})


def media_en_s(df, valor, peso="q", claves=("fuente", "j", "a")):
    # Media de `valor` sobre s ponderada por `peso` (q de cada fuente).  Celdas sin peso
    # quedan fuera.
    d = df[df[peso] > 0]
    num = d.assign(_n=d[valor] * d[peso]).groupby(list(claves))["_n"].sum()
    den = d.groupby(list(claves))[peso].sum()
    return (num / den).rename(valor).reset_index()


def unir(tablas, col="fuente"):
    # Concatena tablas de varias corridas sin repetir una fuente (p. ej. verdad_gill
    # aparece en todas las corridas de Gillingham): se queda la de la primera tabla.
    vistas, out = set(), []
    for t in tablas:
        if t is None:
            continue
        nuevas = [f for f in t[col].unique() if f not in vistas]
        vistas.update(nuevas)
        out.append(t[t[col].isin(nuevas)])
    return pd.concat(out) if out else pd.DataFrame()


def mejores(parametros, rep=None):
    # Estimaciones del mejor arranque (modelo_fin guarda todos; Gillingham solo el mejor)
    d = parametros
    if "mejor" in d:
        d = d[d["mejor"].astype(str).str.lower() == "true"]
    if rep is not None:
        d = d[d["rep"] == rep]
    return d
