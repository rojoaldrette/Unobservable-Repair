# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/theta.py
# Goal:           Parámetros estructurales como pytree dinámico (para estimar)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           06/10/2026
#
# _____________________________________________________________________________
'''
Igual que gillingham/theta.py, adaptado a modelo_fin.

`Params` es estático en jit: cada valor nuevo de un parámetro recompila.  Para estimar:

- `Model(g, th)`: un tipo de hogar.  Junta lo estático de g (tamaños, grid, tolerancias)
  con un dict `th` de arreglos dinámicos.  Es un pytree: `th` son las hojas y `g` el
  dato auxiliar.  Las funciones *_raw del modelo (T_raw, ccps_raw, holding_kernel,
  stationary_q, ...) leen atributos (`m.mu`, `m.u0`, ...), así que aceptan un Params o
  un Model.
- `Economy(g, th, f)`: varios tipos de hogar con fracciones f (observadas, estáticas).
  Los campos de TYPE_FIELDS tienen un primer eje de tipo (mu (T,), u0 (T, J), ...); el
  resto es común.  `eco.type_model(t)` da el Model del tipo t.

`th` trae además `repair_price` (J, A-1): es un dato (la variable excluida), no un
parámetro.  Va en th para que cambiar de año (régimen) no recompile.

Vector libre x (lo que ve el optimizador) <-> th:
    mu, sigma_repair, s_repair, s_sigma = exp(x)   (positivos; s_repair > 0 ordena los
                                                    componentes de la mezcla, Hu & Xin 3(iii))
    sigma_sell                           = sigmoid(x)  (0 < sigma_sell < sigma_trade = 1)
    lo demás                             = x
Sin chatarreo (g.scrap = False), sigma_sell no entra al modelo: estimar.py lo deja fijo.
'''

import numpy as np
import jax
import jax.numpy as jnp

from params import PAPER_TYPES, Types

# Parámetros que se pueden estimar (en este orden aparecen en x)
FIELDS = ("mu", "u0", "u1", "u_s", "tc_buy", "tc_buy_nocar", "tc_sell", "tc_sell_inspect",
          "sigma_sell", "sigma_repair", "s_const", "s_age", "s_persist", "s_repair", "s_sigma")
TYPE_FIELDS = ("mu", "u0", "u1", "tc_buy", "tc_buy_nocar")     # varían por tipo de hogar
TRANSFORM = {"mu": "log", "sigma_repair": "log", "s_repair": "log", "s_sigma": "log",
             "sigma_sell": "logit"}


@jax.tree_util.register_pytree_node_class
class Model:

    def __init__(self, g, th):
        self.g = g
        self.th = th

    def __getattr__(self, name):
        th = self.__dict__["th"]
        if name in th:
            return th[name]
        return getattr(self.__dict__["g"], name)

    def tree_flatten(self):
        keys = tuple(sorted(self.th))
        return tuple(self.th[k] for k in keys), (self.g, keys)

    @classmethod
    def tree_unflatten(cls, aux, children):
        g, keys = aux
        return cls(g, dict(zip(keys, children)))


@jax.tree_util.register_pytree_node_class
class Economy:

    def __init__(self, g, th, f):
        self.g = g
        self.th = th
        self.f = tuple(float(v) for v in f)

    @property
    def n_types(self):
        return len(self.f)

    def type_model(self, t):
        return Model(self.g, {k: (v[t] if k in TYPE_FIELDS else v) for k, v in self.th.items()})

    def with_repair_price(self, R):
        # La misma economía en otro año (otro vector R(j, a))
        return Economy(self.g, dict(self.th, repair_price=jnp.asarray(R, dtype=float)), self.f)

    def tree_flatten(self):
        keys = tuple(sorted(self.th))
        return tuple(self.th[k] for k in keys), (self.g, keys, self.f)

    @classmethod
    def tree_unflatten(cls, aux, children):
        g, keys, f = aux
        return cls(g, dict(zip(keys, children)), f)


# Construcción ______________________________________________________________

def theta_from_g(g):
    th = {k: jnp.asarray(getattr(g, k), dtype=float) for k in FIELDS}
    th["repair_price"] = jnp.asarray(g.repair_price, dtype=float)
    return th


def theta_types(g, types=Types()):
    # th de la economía: comunes de g; por tipo de PAPER_TYPES (primer eje = tipo)
    th = theta_from_g(g)
    for k in TYPE_FIELDS:
        th[k] = jnp.asarray([PAPER_TYPES[name][k] for name in types.names], dtype=float)
    return th


def economy(g, types=Types(), th=None):
    return Economy(g, theta_types(g, types) if th is None else th, types.f)


# Vector libre ______________________________________________________________

def _fwd(name, v):
    t = TRANSFORM.get(name)
    return jnp.exp(v) if t == "log" else jax.nn.sigmoid(v) if t == "logit" else v


def _inv(name, v):
    t = TRANSFORM.get(name)
    return jnp.log(v) if t == "log" else jnp.log(v / (1.0 - v)) if t == "logit" else v


def free_spec(th, free=FIELDS):
    # ((nombre, forma), ...) en el orden de FIELDS: estático y hashable
    return tuple((k, tuple(np.shape(th[k]))) for k in FIELDS if k in free)


def pack(th, spec):
    return jnp.concatenate([jnp.ravel(_inv(k, jnp.asarray(th[k]))) for k, _ in spec])


def unpack(x, spec, th_fixed):
    # th completo: los libres salen de x, el resto (y repair_price) de th_fixed
    th, i = dict(th_fixed), 0
    for k, shape in spec:
        size = int(np.prod(shape))
        th[k] = _fwd(k, x[i:i + size].reshape(shape))
        i += size
    return th


def labels(spec):
    # mu_t0, u0_t1_2 (tipo 1, marca 2), s_const_0 (marca 0), tc_sell, ...
    out = []
    for k, shape in spec:
        for idx in np.ndindex(*shape):
            if k in TYPE_FIELDS:
                out.append("_".join([k, f"t{idx[0]}"] + [str(i) for i in idx[1:]]))
            else:
                out.append("_".join([k] + [str(i) for i in idx]))
    return out


def natural(x, spec, th_fixed):
    # Vector de parámetros en escala natural (mismo orden que labels)
    th = unpack(x, spec, th_fixed)
    return jnp.concatenate([jnp.ravel(th[k]) for k, _ in spec])


def natural_jac_diag(x, spec, th_fixed):
    # d natural / d x (diagonal: cada transformación es elemento a elemento)
    return jnp.diag(jax.jacfwd(lambda x_: natural(x_, spec, th_fixed))(x))
