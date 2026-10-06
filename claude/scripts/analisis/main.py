# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/analisis/main.py
# Goal:           Gráficas y tablas de las estimaciones de modelo_fin y Gillingham
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           06/10/2026
#
# _____________________________________________________________________________
'''
Uso (desde claude/scripts/analisis/):

  python main.py --mf   ../../output/estimaciones/modelo_fin/<tag> \
                 --gill ../../output/estimaciones/gillingham/<tag_datos_modelo_fin> \
                 --gill ../../output/estimaciones/gillingham/<tag_datos_gillingham> \
                 --nombre comparacion_v1

Todo es opcional salvo que haya al menos una corrida.  Escribe en
claude/output/analisis/<nombre>/ las carpetas graficas/ (PNG y PDF) y tablas/ (CSV, TeX, MD).
--regimen: régimen de R para precios (default: el central).  --rep: réplica para las
tablas de parámetros (default: la primera).  --todas: también la tabla completa.
'''

import argparse
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from datos import leer_corrida
import graficas as G
import tablas as Tb

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "output", "analisis"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mf", default="", help="carpeta de una corrida de modelo_fin/estimar.py")
    ap.add_argument("--gill", action="append", default=[], help="carpeta(s) de gillingham/estimar.py")
    ap.add_argument("--nombre", default="analisis")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--regimen", type=int, default=None)
    ap.add_argument("--rep", type=int, default=None)
    ap.add_argument("--todas", action="store_true")
    args = ap.parse_args()

    mf = leer_corrida(args.mf) if args.mf else None
    gills = [leer_corrida(p) for p in args.gill]
    corridas = ([mf] if mf else []) + gills
    if not corridas:
        raise SystemExit("hace falta --mf o --gill")
    tipos = (mf or gills[0]).config["tipos"]["names"]
    out = os.path.join(args.out, args.nombre)
    og, ot = os.path.join(out, "graficas"), os.path.join(out, "tablas")

    G.distribucion_edad(mf, gills, og)
    G.sin_coche(mf, gills, og)
    G.distribucion_s(mf, og)
    G.precios_3d(mf, og, args.regimen)
    G.precios_edad(mf, gills, og, args.regimen)
    G.ccps_edad(mf, gills, og)
    G.mc_sesgo(corridas, og)

    Tb.tabla_parametros(corridas, ot, tipos, args.rep, principales=True)
    Tb.tabla_parametros(corridas, ot, tipos, args.rep, principales=True, sesgo=True)
    if args.todas:
        Tb.tabla_parametros(corridas, ot, tipos, args.rep, principales=False)
        Tb.tabla_parametros(corridas, ot, tipos, args.rep, principales=False, sesgo=True)
    Tb.tabla_mercado(corridas, ot, args.rep)
    Tb.tabla_mc(corridas, ot)
    print(f"listo: {out}")


if __name__ == "__main__":
    main()
