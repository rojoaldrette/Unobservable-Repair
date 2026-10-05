# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/theta.py
# Goal:           Parámetros estructurales como pytree dinámico (para estimar)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           04/10/2026
#
# _____________________________________________________________________________
'''
`GParams` es estático en jit: cada valor nuevo de un parámetro recompila.  Para estimar:

- `GModel(g, th)`: un tipo de hogar.  Junta lo estático de g con un dict `th` de
  parámetros dinámicos.  Pytree: `th` son las hojas y `g` el dato auxiliar (hashable).
  Las funciones *_raw del modelo leen atributos (`m.mu`, ...), así que aceptan un
  GParams o un GModel.
- `Economy(g, th, f)`: T tipos de hogar con fracciones f (observadas, estáticas).
  Los campos de TYPE_FIELDS tienen un primer eje de tipo (mu (T,), u0 (T, J), ...);
  el resto es común (Ts, sigma_sell, accidentes), como en el paper.
  `eco.type_model(t)` da el GModel del tipo t.  Un solo tipo = T = 1.

Vector libre x (lo que ve el optimizador) <-> th:
    mu          = exp(x)        (> 0)
    sigma_sell  = sigmoid(x)    (0 < sigma_sell < sigma_trade = 1, GEV válido)
    lo demás    = x
'''

import numpy as np
import jax
import jax.numpy as jnp

from params import PAPER_TYPES

# Parámetros que se pueden estimar
FIELDS = ("mu", "u0", "u1", "tc_buy", "tc_buy_nocar", "tc_sell", "tc_sell_inspect",
          "sigma_sell", "acc_int", "acc_age")
TYPE_FIELDS = ("mu", "u0", "u1", "tc_buy", "tc_buy_nocar")     # varían por tipo de hogar
TRANSFORM = {"mu": "log", "sigma_sell": "logit"}


@jax.tree_util.register_pytree_node_class
class GModel:

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
        return GModel(self.g, {k: (v[t] if k in TYPE_FIELDS else v) for k, v in self.th.items()})

    def tree_flatten(self):
        keys = tuple(sorted(self.th))
        return tuple(self.th[k] for k in keys), (self.g, keys, self.f)

    @classmethod
    def tree_unflatten(cls, aux, children):
        g, keys, f = aux
        return cls(g, dict(zip(keys, children)), f)


def theta_from_g(g, fields=FIELDS):
    return {k: jnp.asarray(getattr(g, k), dtype=float) for k in fields}


def model(g, th=None):
    return GModel(g, theta_from_g(g) if th is None else th)


def theta_types(g, gt):
    # th de la economía: comunes de g, por tipo de PAPER_TYPES (primer eje = tipo)
    th = theta_from_g(g)
    for k in TYPE_FIELDS:
        th[k] = jnp.asarray([PAPER_TYPES[name][k] for name in gt.names], dtype=float)
    return th


def economy(g, gt, th=None):
    return Economy(g, theta_types(g, gt) if th is None else th, gt.f)


# Vector libre ______________________________________________________________

def _fwd(name, v):
    t = TRANSFORM.get(name)
    return jnp.exp(v) if t == "log" else jax.nn.sigmoid(v) if t == "logit" else v


def _inv(name, v):
    t = TRANSFORM.get(name)
    return jnp.log(v) if t == "log" else jnp.log(v / (1.0 - v)) if t == "logit" else v


def free_spec(th, free=FIELDS):
    # ((nombre, forma), ...) en orden de FIELDS: estático y hashable
    return tuple((k, tuple(np.shape(th[k]))) for k in FIELDS if k in free)


def pack(th, spec):
    return jnp.concatenate([jnp.ravel(_inv(k, jnp.asarray(th[k]))) for k, _ in spec])


def unpack(x, spec, th_fixed):
    # th completo: los libres salen de x, el resto de th_fixed
    th, i = dict(th_fixed), 0
    for k, shape in spec:
        size = int(np.prod(shape))
        th[k] = _fwd(k, x[i:i + size].reshape(shape))
        i += size
    return th


def labels(spec):
    # mu_t0, u0_t1_2 (tipo 1, marca 2), acc_int_0 (marca 0), tc_sell, ...
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
