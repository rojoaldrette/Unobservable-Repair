# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/modelo_tesis/params.py
# Goal:           Configuración y parámetros del modelo de la tesis (docs/modelo_fin.md)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           08/10/2026
#
# _____________________________________________________________________________
'''
Especificación: claude/niu/docs/modelo_fin.md.  Aquí solo los números.

- `Config` (estático, hashable; entra a jit como argumento estático): marcas, tipos,
  a_max, beta, escalas GEV fijas, grid de w, regímenes de R y tolerancias.
- `theta(cfg)` (dinámico, dict de arreglos): parámetros estructurales.  Los de Gillingham
  vienen de las tablas del paper (se leen de ../gillingham/params.py, una sola fuente);
  los de desgaste y reparación son la calibración aprobada (modelo_fin.md, secs. 3-4).
- `regime_theta(th, cfg, t)`: el θ del régimen t (R_t = R e^{ζ_t}).

El grid de w es estático (no cambia con σ_η al estimar): w_k = h k, con 0 en el grid
(ahí nacen los coches nuevos).
'''

import importlib.util
import os
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

# Tablas de Gillingham: una sola fuente (niu/gillingham/params.py)
_spec = importlib.util.spec_from_file_location(
    "gill_params", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gillingham", "params.py"))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

TYPE_KEYS = G.TYPE_KEYS          # campos con primer eje = tipo de hogar


@dataclass(frozen=True)
class Config:

    # Economía (como niu/gillingham)
    brands: tuple = ("light_brown", "heavy_brown")
    types: tuple = ("low_couple_poor", "low_single_poor")
    f: tuple = (0.5, 0.5)
    a_max: int = 25
    beta: float = 0.95
    p_scrap: float = 2.0
    inspect_age_min: int = 4
    nocar: str = "reemplaza"

    # Escalas GEV fijas: sigma >= sigma_trade >= sigma_w (nido de w al comprar)
    sigma: float = 1.0
    sigma_trade: float = 1.0
    sigma_w: float = 0.1

    # Grid de w: paso h = sigma_eta / 2 de la calibración; [w_min, w_max] múltiplos de h
    w_step: float = 0.075
    w_min: float = -3.0
    w_max: float = 2.25
    eps_F: float = 1e-10          # piso de la transición de w: ninguna celda con masa 0 exacta

    # Regímenes de R (variable excluida)
    zetas: tuple = (-0.3, 0.0, 0.3)

    # Solver
    sa_iter: int = 500
    eq_tol: float = 1e-9
    eq_max_iter: int = 60
    jac_chunk: int = 256          # columnas del jacobiano por bloque (memoria en GPU)

    def __post_init__(self):
        assert all(b in G.BRANDS for b in self.brands) and all(t in G.TYPES for t in self.types)
        assert len(self.f) == len(self.types) and abs(sum(self.f) - 1) < 1e-12
        assert 0 < self.sigma_w <= self.sigma_trade <= self.sigma
        for v in (self.w_min, self.w_max):
            assert abs(v / self.w_step - round(v / self.w_step)) < 1e-9, "w_min, w_max deben ser múltiplos de w_step"
        assert self.w_min < 0 < self.w_max

    @property
    def J(self):
        return len(self.brands)

    @property
    def T(self):
        return len(self.types)

    @property
    def w_grid(self):
        k0, k1 = round(self.w_min / self.w_step), round(self.w_max / self.w_step)
        return self.w_step * np.arange(k0, k1 + 1)

    @property
    def W(self):
        return len(self.w_grid)

    @property
    def w0(self):
        # índice de w = 0 (coche nuevo)
        return int(np.argmin(np.abs(self.w_grid)))


# Calibración aprobada de desgaste y reparación (modelo_fin.md, secs. 3.2-4.2)
SIGMA_ETA = 0.15
KAPPA = 0.25
P_BAR = 0.3                       # tasa de reparación objetivo: delta_j = acc_age_j + kappa p_bar
U_W = -0.2                        # utils por unidad de w (propuesta; el autor probará 0)
SIGMA_REP = 0.3
# R(j, a) = base + pendiente (a - 1) + prima e^{-(a-1)/tau}, miles de DKK.  La prima de
# coches jóvenes es el costo de piezas nuevas (menos disponibles).  (base, pendiente, prima, tau)
R_BASE = {"light_brown": (4.0, 0.10, 6.0, 4.0), "light_green": (3.5, 0.21, 0.0, 1.0),
          "heavy_brown": (10.0, 0.60, 0.0, 1.0), "heavy_green": (8.0, 0.48, 0.0, 1.0)}


def repair_price(brand, a):
    base, slope, prima, tau = R_BASE[brand]
    return base + slope * (a - 1) + prima * np.exp(-(a - 1) / tau)


def theta(cfg, **over):
    # θ completo.  `over` reemplaza valores (p. ej. theta(cfg, u_w=0.0)).
    jb = [G.BRANDS.index(b) for b in cfg.brands]
    tt = [G.TYPES[t] for t in cfg.types]
    pick = lambda v: np.asarray(v, float)[jb]
    acc_age = pick(G.ACC_AGE)
    a = np.arange(1, cfg.a_max)
    th = dict(
        # Gillingham
        mu=[t["mu"] for t in tt],
        u0=[pick(t["u0"]) for t in tt],
        u1=[pick(t["u1"]) for t in tt],
        u2=np.zeros((cfg.T, cfg.J)),
        tc_buy=[t["tc_buy"] for t in tt],
        tc_buy_nocar=[t["tc_buy_nocar"] for t in tt],
        tc_sell=G.TC_SELL, tc_sell_inspect=G.TC_SELL_INSPECT, sigma_sell=G.SIGMA_SELL,
        acc_int=pick(G.ACC_INT), acc_age=acc_age,
        p_new=pick(G.P_NEW), p_scrap=np.full(cfg.J, cfg.p_scrap),
        # desgaste y reparación
        u_w=U_W, delta=acc_age + KAPPA * P_BAR, kappa=KAPPA, sigma_eta=SIGMA_ETA,
        sigma_rep=SIGMA_REP,
        R=np.stack([repair_price(b, a) for b in cfg.brands]),                     # (J, A-1)
    )
    th.update(over)
    return {k: jnp.asarray(v, dtype=float) for k, v in th.items()}


def regime_theta(th, cfg, t):
    return dict(th, R=th["R"] * np.exp(cfg.zetas[t]))
