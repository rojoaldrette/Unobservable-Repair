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
`GParams` es estático en jit: cada valor nuevo de un parámetro recompila.  Para estimar,
`GModel(g, th)` junta lo estático (dimensiones, solvers, p_new, p_scrap, beta...) con un
dict `th` de parámetros dinámicos.  Se registra como pytree: `th` son las hojas y `g`
el dato auxiliar (hashable), así que jit recompila solo si cambia `g`.

Todas las funciones *_raw del modelo leen atributos (`m.mu`, `m.u0`, ...), así que
aceptan indistintamente un GParams o un GModel.

Vector libre x (lo que ve el optimizador) <-> th:
    mu          = exp(x)        (> 0)
    sigma_sell  = sigmoid(x)    (0 < sigma_sell < sigma_trade = 1, GEV válido)
    lo demás    = x
'''

import numpy as np
import jax
import jax.numpy as jnp

# Parámetros que se pueden estimar (los 18 del hogar de la réplica)
FIELDS = ("mu", "u0", "u1", "tc_buy", "tc_buy_nocar", "tc_sell", "tc_sell_inspect",
          "sigma_sell", "acc_int", "acc_age")
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


def theta_from_g(g, fields=FIELDS):
    return {k: jnp.asarray(getattr(g, k), dtype=float) for k in fields}


def model(g, th=None):
    return GModel(g, theta_from_g(g) if th is None else th)


# Vector libre ______________________________________________________________

def _fwd(name, v):
    t = TRANSFORM.get(name)
    return jnp.exp(v) if t == "log" else jax.nn.sigmoid(v) if t == "logit" else v


def _inv(name, v):
    t = TRANSFORM.get(name)
    return jnp.log(v) if t == "log" else jnp.log(v / (1.0 - v)) if t == "logit" else v


def free_spec(g, free=FIELDS):
    # ((nombre, tamaño), ...) en orden de FIELDS: static y hashable
    return tuple((k, int(np.size(getattr(g, k)))) for k in FIELDS if k in free)


def pack(th, spec):
    return jnp.concatenate([jnp.ravel(_inv(k, jnp.asarray(th[k]))) for k, _ in spec])


def unpack(x, spec, th_fixed):
    # th completo: los libres salen de x, el resto de th_fixed
    th, i = dict(th_fixed), 0
    for k, size in spec:
        v = x[i:i + size]
        th[k] = _fwd(k, v if np.ndim(th_fixed[k]) else v[0])
        i += size
    return th


def labels(spec):
    out = []
    for k, size in spec:
        out += [k] if size == 1 else [f"{k}_{j}" for j in range(size)]
    return out


def natural(x, spec, th_fixed):
    # Vector de parámetros en escala natural (mismo orden que labels)
    th = unpack(x, spec, th_fixed)
    return jnp.concatenate([jnp.ravel(th[k]) for k, _ in spec])


def natural_jac_diag(x, spec, th_fixed):
    # d natural / d x (diagonal: cada transformación es elemento a elemento)
    return jnp.diag(jax.jacfwd(lambda x_: natural(x_, spec, th_fixed))(x))
