# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/gillingham/params.py
# Goal:           Estimaciones publicadas de Gillingham et al. y configuración
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           07/10/2026
#
# _____________________________________________________________________________
'''
Gillingham, Iskhakov, Munk-Nielsen, Rust & Schjerning (2019 WP), "Equilibrium Trade in
Automobiles".  Todos los números salen del apéndice F (Tablas 3, 4, 5, 7, 8, 9, 10).

Unidades: precios en miles de DKK; utilidades en utils; mu = utils por mil DKK
(ej.: u0 / mu = disposición a pagar por un año de coche nuevo; 3.649 / 0.1131 = 32.3 mil
DKK, el número que da el texto de la sec. 6.2 para "Low WD, Couple, Poor").

Dos objetos:
- `Config` (estático, hashable): qué marcas y tipos de hogar, a_max, beta, escalas GEV,
  lecturas de las tablas y tolerancias.  Entra a jit como argumento estático.
- `theta(cfg)` (dinámico, dict de arreglos): los parámetros estructurales.  Es lo que se
  estima.  Los campos de TYPE_KEYS tienen primer eje = tipo de hogar.

Lo que el paper NO reporta (y aquí se fija):
- u2 (término cuadrático en la edad): 0.
- dummy de edad par en la utilidad (sec. 6.2): no está en las tablas; se omite.
- escalas de los nidos de marca y edad: 1 (solo reportan la de vender/chatarrear).
- p_scrap y beta: 2 mil DKK y 0.95.
- fracciones f de los tipos: mitad y mitad.
'''

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

# Tablas del paper _____________________________________________________________

BRANDS = ("light_brown", "light_green", "heavy_brown", "heavy_green")

P_NEW = (142.33, 110.09, 275.84, 205.25)       # Tabla 3, precio nuevo (miles DKK)
ACC_INT = (-5.6248, -6.0443, -5.6728, -5.7826)  # Tabla 4, logit de accidentes: intercepto
ACC_AGE = (0.1804, 0.2216, 0.2020, 0.2048)      # Tabla 4, pendiente en la edad

# Tabla 5 (leída con pdftotext -raw; las etiquetas del modo -layout se desalinean).
# Los errores estándar de la tabla son idénticos a las estimaciones: hay un problema de
# formato en el WP.  Lectura usada: escala del nido vender/chatarrear, Ts normal y Ts en
# año de inspección (el −2.1929 se lee como un costo de 2.1929 en la utilidad de vender:
# así sale el zig-zag de chatarreo de la figura 7).
SIGMA_SELL = 0.3454
TC_SELL = 0.9106
TC_SELL_INSPECT = 2.1929

# Tablas 7 (mu), 8 (u0), 9 (u1) y 10 (Tb "common" y "no car"); u0 y u1 en el orden de BRANDS
TYPES = {
    "low_couple_poor":  dict(mu=0.1131, u0=(3.6490, 3.1132, 5.1535, 4.7760),
                             u1=(-0.1459, -0.0922, -0.2196, -0.1717), tc_buy=6.5944, tc_buy_nocar=1.7899),
    "low_couple_rich":  dict(mu=0.1119, u0=(4.0324, 3.4657, 5.7492, 5.3478),
                             u1=(-0.1586, -0.0985, -0.2411, -0.1953), tc_buy=6.4425, tc_buy_nocar=1.0719),
    "low_single_poor":  dict(mu=0.0941, u0=(2.4042, 2.1504, 3.5823, 3.2068),
                             u1=(-0.0984, -0.0615, -0.1600, -0.1128), tc_buy=6.5457, tc_buy_nocar=3.0816),
    "low_single_rich":  dict(mu=0.1077, u0=(3.2454, 2.8222, 4.6934, 4.2829),
                             u1=(-0.1308, -0.0841, -0.2056, -0.1544), tc_buy=6.6036, tc_buy_nocar=2.5769),
    "high_couple_poor": dict(mu=0.1036, u0=(3.8821, 3.4351, 5.3199, 5.0561),
                             u1=(-0.1390, -0.0825, -0.2134, -0.1696), tc_buy=6.2644, tc_buy_nocar=0.7793),
    "high_couple_rich": dict(mu=0.1155, u0=(4.7620, 4.3185, 6.5617, 6.2375),
                             u1=(-0.1642, -0.1069, -0.2551, -0.2074), tc_buy=6.4237, tc_buy_nocar=0.1742),
    "high_single_poor": dict(mu=0.0920, u0=(2.6685, 2.4290, 3.7554, 3.5129),
                             u1=(-0.1108, -0.0707, -0.1732, -0.1273), tc_buy=6.0303, tc_buy_nocar=2.3383),
    "high_single_rich": dict(mu=0.1081, u0=(3.5538, 3.1741, 4.9946, 4.6985),
                             u1=(-0.1493, -0.1007, -0.2208, -0.1735), tc_buy=6.2786, tc_buy_nocar=1.7884),
}

# Campos de theta que varían por tipo de hogar (primer eje); el resto es común
TYPE_KEYS = ("mu", "u0", "u1", "u2", "tc_buy", "tc_buy_nocar")


# Configuración ______________________________________________________________

@dataclass(frozen=True)
class Config:

    # Economía
    brands: tuple = ("light_brown", "heavy_brown")    # barato y caro; ver docs/gillingham.md
    types: tuple = ("low_couple_poor", "low_single_poor")
    f: tuple = (0.5, 0.5)                 # fracción de cada tipo (observada; no la reporta el paper)
    a_max: int = 25                       # edades 0..24; a_max es la terminal (chatarreo forzado)
    beta: float = 0.95
    p_scrap: float = 2.0                  # miles de DKK
    inspect_age_min: int = 4              # inspección en edades pares >= 4

    # Escalas GEV: sigma (purge | keep | trade) >= sigma_trade (qué comprar) >= sigma_sell
    sigma: float = 1.0
    sigma_trade: float = 1.0

    # Lecturas de las tablas
    tc_units: str = "utils"               # "utils": Tb, Ts restan utils; "dkk": restan mu * T
    nocar: str = "reemplaza"              # sin coche paga solo Tb_nocar ("reemplaza") o Tb + Tb_nocar ("suma")

    # Solvers
    sa_iter: int = 1_000                  # aproximaciones sucesivas al arrancar (error ~ beta^1000)
    eq_tol: float = 1e-10                 # max |F| del sistema conjunto (Bellman + mercado)
    eq_max_iter: int = 50

    def __post_init__(self):
        assert all(b in BRANDS for b in self.brands), self.brands
        assert all(t in TYPES for t in self.types), self.types
        assert len(self.f) == len(self.types) and abs(sum(self.f) - 1) < 1e-12
        assert self.tc_units in ("utils", "dkk") and self.nocar in ("suma", "reemplaza")
        assert 0 < SIGMA_SELL <= self.sigma_trade <= self.sigma

    @property
    def J(self):
        return len(self.brands)

    @property
    def T(self):
        return len(self.types)


def theta(cfg):
    # Parámetros estructurales del paper para las marcas y tipos de cfg
    jb = [BRANDS.index(b) for b in cfg.brands]
    tt = [TYPES[t] for t in cfg.types]
    pick = lambda v: np.asarray(v, float)[jb]
    th = dict(
        mu=[t["mu"] for t in tt],
        u0=[pick(t["u0"]) for t in tt],
        u1=[pick(t["u1"]) for t in tt],
        u2=np.zeros((cfg.T, cfg.J)),
        tc_buy=[t["tc_buy"] for t in tt],
        tc_buy_nocar=[t["tc_buy_nocar"] for t in tt],
        tc_sell=TC_SELL,
        tc_sell_inspect=TC_SELL_INSPECT,
        sigma_sell=SIGMA_SELL,
        acc_int=pick(ACC_INT),
        acc_age=pick(ACC_AGE),
        p_new=pick(P_NEW),
        p_scrap=np.full(cfg.J, cfg.p_scrap),
    )
    return {k: jnp.asarray(v, dtype=float) for k, v in th.items()}
