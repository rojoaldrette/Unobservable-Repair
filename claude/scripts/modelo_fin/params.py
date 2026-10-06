# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/params.py
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
- Costos de transacción como en Gillingham (sec. 3.1): aditivamente separables,
  T = Tb(j, d) + Ts(i, a), en utils.
  * Comprador (Tabla 10): "common" = tc_buy para toda compra; "no car" = tc_buy_nocar
    extra si compra desde el estado sin coche.
  * Vendedor (Tabla 5, leída con el texto en modo raw; ver docs/gillingham.md):
        lambda_s (escala vender/chatarrear)          0.3454  -> no se usa (no hay chatarreo)
        sales transaction cost                       0.9106  -> tc_sell
        sales transaction cost (inspection year)    -2.1929  -> coeficiente en la utilidad
                                                                de vender: tc_sell_inspect = 2.1929
    Años de inspección: edades pares >= 4 (inspección bianual en Dinamarca).
    v1.0 usaba 0.3454 / 0.9106 (lectura con -layout, desalineada).  Por confirmar
    con la versión publicada o el código de los autores.
- Sin chatarreo endógeno: un coche activo solo sale de tus manos vendiéndolo (trade
  o purge), por accidente o al llegar a la edad terminal.  p_scrap solo lo cobra el
  dueño de un coche terminal.
- beta no se reporta en la aplicación empírica; 0.95 es el del ejemplo §4.3.
- Precios de chatarra, reparación y toda la dinámica de s son inventados (tuyos).
'''

from dataclasses import dataclass, field


def _default_repair_prices():
    # R(j, a) en miles de DKK para a = 1..a_max-1 (aquí a_max = 7 -> 6 edades)
    # Calibrado para que el shock logit pese poco: con sigma_repair = 1 la prob. de
    # reparar "por puro ruido" (sin beneficio) es 1/(1+exp(mu R)) ~ 5-15%.
    # (Antes base = (3.0, 2.5, 5.0): el ruido daba ~40% y dominaba la decisión.)
    base = (15.0, 12.5, 25.0)
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
    tc_buy: float = 6.5944      # Tb: comprador, cualquier compra (Tabla 10, "common")
    tc_buy_nocar: float = 1.7899  # extra si compra desde "sin coche" (Tabla 10, "no car")
    tc_sell: float = 0.9106     # Ts: vendedor, año sin inspección (Tabla 5, raw)
    tc_sell_inspect: float = 2.1929  # Ts: vendedor, año de inspección (Tabla 5, raw: -2.1929 en utilidad)
    inspect_age_min: int = 4    # inspección en edades pares >= inspect_age_min

    # Precios exógenos (miles de DKK)
    p_new: tuple = (142.33, 110.09, 275.84)
    p_scrap: tuple = (2.0, 2.0, 2.0)   # solo coches terminales (accidente o edad a_max)
    # Precio de reparación R(j, a), a = 1..a_max-1.  Es tu variable excluida
    # (Hu & Xin, Assumption 7): mueve las CCPs pero no la transición de s.
    # Para otro "año" usa dataclasses.replace(g, repair_price=...).
    repair_price: tuple = field(default_factory=_default_repair_prices)


    # Shocks (valor extremo tipo I) #####################
    # Decisión secuencial dentro del periodo:
    #   1) {keep, purge, trade}; dentro de trade, qué coche h comprar.
    #   2) Ya con el coche h en la mano, se realiza un shock nuevo y se decide
    #      reparar o no.  Pr(repair | h) no depende de cómo llegaste a h.
    # sigma == sigma_trade  ->  logit simple en la etapa 1.  Para nested logit:
    # 0 < sigma_trade <= sigma.  sigma_repair es la escala del shock de la etapa 2;
    # como se realiza después, no necesita ordenarse respecto a los otros.
    sigma: float = 1.0
    sigma_trade: float = 1.0
    sigma_repair: float = 1.0
    # Nido de s dentro de cada (j, d) al comprar.  Si sigma_s == sigma_trade cada celda
    # (j, d, s) es una alternativa logit con su propio shock y el modelo depende del
    # tamaño del grid (más celdas = más "variedad" ficticia).  Con sigma_s chico el
    # comprador elige s casi por valor neto (precio hedónico) y la dependencia del grid
    # se va a cero (el valor inclusivo crece como sigma_s * log n_s).
    sigma_s: float = 0.1


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

    # Solver de ED(P) = 0 (ver ED.py).  Se resuelve en logs: log D - log S = 0.
    ed_floor: float = 1e-10     # piso para S: q por debajo de ~1e-13 es ruido numérico del solve
    ed_tol: float = 1e-8        # max |log D - log S| en celdas con masa
    ed_mass_tol: float = 1e-9   # una celda "tiene masa" si S o D > ed_mass_tol
    tat_max_iter: int = 200     # tâtonnement con jacobiano diagonal (primera fase)
    tat_tol: float = 0.1        # ...hasta aquí, luego Newton-Krylov (converge en ~3 pasos desde 2e-2)
    tat_damp: float = 0.5
    nk_ed_max_iter: int = 30    # pasos de Newton-Krylov sobre P
    gmres_tol: float = 1e-10
    gmres_restart: int = 60     # dimensión del subespacio de Krylov antes de reiniciar
    gmres_maxiter: int = 5      # reinicios de GMRES (también en los pasos de Newton de la Bellman)


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
        if not (0 < self.sigma_s <= self.sigma_trade <= self.sigma):
            raise ValueError("GEV válido requiere 0 < sigma_s <= sigma_trade <= sigma")
        if not self.sigma_repair > 0:
            raise ValueError("sigma_repair > 0")
        if A < 3:
            raise ValueError("a_max >= 3")
