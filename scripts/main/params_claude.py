# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         scripts/params.py
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________
'''
Convenciones
------------
- Un periodo = un año.  Precios en miles de DKK, utilidades en "utils".
- Marcas j = 0..J-1.  Edades a = 1..a_max-1 son "activas"; a = a_max es terminal
  (el coche se chatarrea exógenamente al inicio del periodo, s ya no importa).
- s = probabilidad de descomponerse (pérdida total) durante el periodo.  Vive en
  un grid de n_s puntos en [s_min, s_max] ⊂ [0, 1].
- Todos los "vectores" son tuplas para que Params sea hashable y se pueda pasar
  como argumento estático a jax.jit (static_argnames="g").

Valores por defecto (Gillingham, Iskhakov, Munk-Nielsen, Rust & Schjerning):
- Hogar tipo "Low WD, Couple, Poor" (Tablas 7–10 del apéndice).
- Marcas: (light brown, light green, heavy brown); heavy green se omite para J=3.
- u2 (término cuadrático en edad) no se reporta en las tablas -> 0.
- Tabla 10: "common" se usa como costo de transacción del comprador (utils) y
  "no car" como costo adicional al comprar desde el estado sin coche.  Esa es
  mi lectura de la tabla; verifícala con el texto/código de los autores.
- beta no se reporta en la aplicación empírica; 0.95 es el del ejemplo §4.3.
- Precios de chatarra, reparación y toda la dinámica de s son inventados (tuyos).
'''

from dataclasses import dataclass, field


def _default_repair_prices():
    # R(j, a) en miles de DKK para a = 1..a_max-1 (aquí a_max = 7 -> 6 edades)
    base = (3.0, 2.5, 5.0)
    growth = 0.15
    return tuple(tuple(round(b * (1 + growth * (a - 1)), 4) for a in range(1, 7))
                 for b in base)


@dataclass(frozen=True)
class Params:

    # Structural params ###################

    beta: float = 0.95

    # Utilidad marginal del dinero (utils por mil DKK)
    mu: float = 0.1131

    # Utilidad de flujo de tener un coche (j, a, s):
    #   u0_j + u1_j * a + u2_j * a^2 + u_s * s      (a = 0 es coche nuevo)
    u0: tuple = (3.6490, 3.1132, 5.1535)
    u1: tuple = (-0.1459, -0.0922, -0.2196)
    u2: tuple = (0.0, 0.0, 0.0)
    u_s: float = -2.0           # desutilidad de manejar un coche "frágil"
    u_none: float = 0.0         # utilidad de no tener coche (normalización)

    # Costos de transacción (utils)
    tc_buy: float = 6.5944      # comprador, cualquier compra
    tc_buy_nocar: float = 1.7899  # extra si compra desde "sin coche"
    tc_sell: float = 0.9106     # vendedor en el mercado secundario (chatarrear = 0)

    # Precios exógenos (miles de DKK)
    p_new: tuple = (142.33, 110.09, 275.84)
    p_scrap: tuple = (2.0, 2.0, 2.0)
    # Precio de reparación R(j, a), a = 1..a_max-1.  Es tu variable excluida
    # (Hu & Xin, Assumption 7): mueve las CCPs pero no la transición de s.
    # Para otro "año" usa dataclasses.replace(g, repair_price=...).
    repair_price: tuple = field(default_factory=_default_repair_prices)


    # Shocks (valor extremo tipo I) #####################
    # Árbol: arriba {keep, repair, purge, trade}; dentro de trade el coche (j,d,s);
    # dentro de vender-o-chatarrear el canal de salida.
    # sigma == sigma_trade == sigma_sell  ->  logit multinomial simple (EV1).
    # Para nested logit basta con 0 < sigma_sell <= sigma_trade <= sigma.
    sigma: float = 1.0
    sigma_trade: float = 1.0
    sigma_sell: float = 1.0


    # State variables #####################

    n_brands: int = 3
    a_max: int = 7              # edad terminal

    # Grid de s (probabilidad de descomponerse)
    s_min: float = 0.0
    s_max: float = 0.5
    n_s: int = 100
    s_new: float = 0.03         # todos los coches nuevos empiezan aquí (se ajusta al grid)

    # Transición de s  (Hu & Xin, Assumption 2):
    #   s' = m(r, j, a, s) + eta,   eta ~ N(0, s_sigma^2), discretizado en el grid
    #   m  = s_const_j + s_age * a + s_persist * s - s_repair * r
    # Solo aplica si el coche sobrevive (prob 1 - s); si se descompone va a a_max.
    s_const: tuple = (0.020, 0.015, 0.025)
    s_age: float = 0.005
    s_persist: float = 1.0
    s_repair: float = 0.06
    s_sigma: float = 0.015


    # Equilibrio #########################

    dep_factor: float = 0.87    # para el precio inicial P0 = p_new * dep^a * (1 - s)


    # Value function iteration #################################

    sa_tol: float = 1e-6        # successive approximations hasta aquí...
    sa_max_iter: int = 5_000
    vfi_tol: float = 1e-12      # ...luego Newton-Kantorovich hasta aquí
    nk_max_iter: int = 20


    # Monte Carlo ########################################

    mc_replic: int = 250


    def __post_init__(self):
        J, A = self.n_brands, self.a_max
        for name in ("u0", "u1", "u2", "p_new", "p_scrap", "s_const"):
            if len(getattr(self, name)) != J:
                raise ValueError(f"{name} debe tener longitud n_brands={J}")
        if len(self.repair_price) != J or any(len(r) != A - 1 for r in self.repair_price):
            raise ValueError(f"repair_price debe ser J x (a_max-1) = {J} x {A - 1}")
        if not (0.0 <= self.s_min < self.s_max <= 1.0):
            raise ValueError("el grid de s debe estar en [0, 1]")
        if not (0 < self.sigma_sell <= self.sigma_trade <= self.sigma):
            raise ValueError("GEV válido requiere 0 < sigma_sell <= sigma_trade <= sigma")
        if A < 3:
            raise ValueError("a_max >= 3")
