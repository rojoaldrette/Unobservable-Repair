# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/calibracion.py
# Goal:           Calibraciones con nombre (la de la tesis: "tesis")
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           06/10/2026
#
# _____________________________________________________________________________
'''
`calibracion(nombre, a_max, n_s)` -> Params.  La que se usa para correr es "tesis".

"tesis" (Fase 3; default de estimar.py, main.py y compare.py)
    Mismos parámetros que Gillingham, para que la única diferencia sea poder reparar:
    - a_max = 25 (el del paper), dos tipos de hogar en estimar.py (Tablas 7-10).
    - u0, u1, mu, tc_buy, tc_buy_nocar, tc_sell, tc_sell_inspect: Tablas 5, 7-10
      (los defaults de Params, que ya son los de Gillingham).
    - u_s = 0: Gillingham no tiene desutilidad de un coche frágil.
    - Desgaste en log-odds, ℓ = logit(s):
          ℓ_nuevo = acc_int_j              (Tabla 4: logit de accidente en edad 0)
          ℓ'      = ℓ + acc_age_j − s_repair r + η,   η ~ N(0, s_sigma²)
      Sin reparar y sin ruido, s = sigmoid(acc_int_j + acc_age_j a): exactamente la
      logit de accidentes de Gillingham.  Por eso s_const_j = acc_age_j, s_age = 0,
      s_persist = 1.
    - Grid de ℓ en [−9, 1] con 100 puntos (paso 0.1): s de 1e-4 a 0.73.  ℓ de un coche
      nuevo ≈ −6; a los 24 años sin reparar ≈ −1 (s ≈ 0.2-0.33).
    - s_repair = 0.5: reparar baja las odds de descomponerse ~40% (e^-0.5 = 0.61).
      s_sigma = 0.15.  Propios (no hay dato); ver docs/manual.md.
    - R(j, a) = (4, 6.5, 10) mil DKK en a = 1, +6% lineal por año: ligero gasolina bajo,
      ligero diésel medio, pesado gasolina alto.
    - sigma_repair = 0.3: con R realista, mu R pesa 0.5-1.1 utils; con sigma_repair = 1
      el ruido dominaría la decisión de reparar.
    - Chatarreo endógeno encendido (scrap = True, sigma_sell = 0.3454, Tabla 5), como en
      Gillingham.  Sin él, con a_max = 25 los coches viejos solo salen vendiéndose y sus
      precios quedan negativos (visto en la prueba del 2026-10-06).

Las anteriores (para reproducir resultados viejos):
    "defaults"  Params() con a_max y n_s dados (s en niveles, accidentes de 3-20%)
    "gill_s"    s en niveles calibrada a la Tabla 4 en edades bajas (M7_gill de compare.py)
    "tesis_v0"  gill_s + R realista + sigma_repair = 0.3 (default de v1.7)
'''

import dataclasses

from params import Params

# Tabla 4 de Gillingham: logit de accidentes, marcas LB, LG, HB
ACC_INT = (-5.6248, -6.0443, -5.6728)
ACC_AGE = (0.1804, 0.2216, 0.2020)

NOMBRES = ("tesis", "tesis_v0", "gill_s", "defaults")


def repair_prices(a_max, base, growth):
    # R(j, a) = base_j (1 + growth (a − 1)),  a = 1..a_max−1
    return tuple(tuple(round(b * (1 + growth * (a - 1)), 4) for a in range(1, a_max)) for b in base)


def calibracion(nombre="tesis", a_max=25, n_s=100):
    if nombre == "tesis":
        return dataclasses.replace(
            Params(), a_max=a_max, n_s=n_s,
            u_s=0.0,
            s_space="logodds", s_min=-9.0, s_max=1.0, s_new=ACC_INT,
            s_const=ACC_AGE, s_age=0.0, s_persist=1.0, s_repair=0.5, s_sigma=0.15,
            repair_price=repair_prices(a_max, (4.0, 6.5, 10.0), 0.06),
            sigma_repair=0.3,
            scrap=True, sigma_sell=0.3454)
    g = dataclasses.replace(Params(), a_max=a_max, n_s=n_s,
                            repair_price=repair_prices(a_max, (15.0, 12.5, 25.0), 0.15))
    if nombre == "defaults":
        return g
    g = dataclasses.replace(g, s_max=0.1, s_new=0.004, s_const=(0.002, 0.002, 0.002),
                            s_age=0.0003, s_sigma=0.003, s_repair=0.004)
    if nombre == "gill_s":
        return g
    if nombre == "tesis_v0":
        return dataclasses.replace(g, repair_price=repair_prices(a_max, (4.0, 6.5, 10.0), 0.06),
                                   sigma_repair=0.3)
    raise ValueError(f"calibración desconocida: {nombre} (opciones: {NOMBRES})")
