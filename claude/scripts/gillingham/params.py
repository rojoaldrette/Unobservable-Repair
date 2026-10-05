# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/params.py
# Goal:           Parámetros de la réplica de Gillingham, Iskhakov, Munk-Nielsen,
#                 Rust & Schjerning (2019 WP, "Equilibrium Trade in Automobiles")
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
Modelo de Gillingham et al. (sec. 3 y 6), un tipo de hogar, estado (j, a):
- Elecciones del dueño de (i, a), a < a_max: keep | purge | trade a (j, d).
  Al deshacerse del coche: vender (paga Ts) o chatarrear (cobra p_scrap, Ts = 0),
  nido con escala sigma_sell (ec. 18: decisión estática, misma para purge y trade).
- Accidente: prob alpha(j, a) = logit(acc_int_j + acc_age_j * a)  (Tabla 4).
- a = a_max: chatarreo forzado; el dueño cobra p_scrap y elige como si no tuviera coche.

Fuentes de los números (hogar "Low WD, Couple, Poor"; marcas LB, LG, HB):
- mu, u0, u1: Tablas 7, 8, 9.  u2 no se reporta -> 0.
- tc_buy ("common"), tc_buy_nocar ("no car"): Tabla 10.
- Tabla 5, leída con el texto en modo raw (la versión -layout desalinea las etiquetas):
      lambda_s (escala vender/chatarrear)          0.3454  -> sigma_sell
      sales transaction cost                       0.9106  -> tc_sell
      sales transaction cost (inspection year)    -2.1929  -> coeficiente en la utilidad
                                                              de vender: costo 2.1929
  INTERPRETACIÓN POR VERIFICAR (los errores estándar de la tabla son iguales a las
  estimaciones, señal de un problema de formato).  Con esta lectura vender es más caro
  en años de inspección -> más chatarreo en edades pares (el zig-zag).
- El dummy de edad par en la utilidad (sec. 6.2) no aparece en las tablas -> u_even = 0.
- p_new: Tabla 3.  p_scrap y beta no se reportan: 2.0 y 0.95 (mismos que modelo_fin).
- a_max = 25 en el paper (edades 0..24).  Para comparar con modelo_fin se usa 7.
'''

from dataclasses import dataclass


@dataclass(frozen=True)
class GParams:

    beta: float = 0.95
    mu: float = 0.1131

    # u(j, a) = u0_j + u1_j a + u2_j a^2 + u_even * 1{a par, a >= inspect_age_min}
    u0: tuple = (3.6490, 3.1132, 5.1535)
    u1: tuple = (-0.1459, -0.0922, -0.2196)
    u2: tuple = (0.0, 0.0, 0.0)
    u_even: float = 0.0
    u_none: float = 0.0

    # Costos de transacción (utils)
    tc_buy: float = 6.5944
    tc_buy_nocar: float = 1.7899
    tc_sell: float = 0.9106
    tc_sell_inspect: float = 2.1929
    inspect_age_min: int = 4
    term_pays_nocar: bool = False   # ¿el dueño de un terminal paga tc_buy_nocar? (igual que modelo_fin)

    # Precios (miles de DKK)
    p_new: tuple = (142.33, 110.09, 275.84)
    p_scrap: tuple = (2.0, 2.0, 2.0)

    # Accidentes (Tabla 4): logit(acc_int + acc_age * a)
    acc_int: tuple = (-5.6248, -6.0443, -5.6728)
    acc_age: tuple = (0.1804, 0.2216, 0.2020)

    # Shocks: sigma >= sigma_trade >= sigma_sell
    sigma: float = 1.0
    sigma_trade: float = 1.0
    sigma_sell: float = 0.3454

    n_brands: int = 3
    a_max: int = 25

    dep_factor: float = 0.87

    # Solvers
    sa_tol: float = 1e-6
    sa_max_iter: int = 5_000
    vfi_tol: float = 1e-12
    nk_max_iter: int = 20
    ed_tol: float = 1e-10
    ed_max_iter: int = 50

    def __post_init__(self):
        J = self.n_brands
        for name in ("u0", "u1", "u2", "p_new", "p_scrap", "acc_int", "acc_age"):
            if len(getattr(self, name)) != J:
                raise ValueError(f"{name} debe tener longitud n_brands={J}")
        if not (0 < self.sigma_sell <= self.sigma_trade <= self.sigma):
            raise ValueError("GEV válido requiere 0 < sigma_sell <= sigma_trade <= sigma")
