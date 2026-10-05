# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/modelo_fin/probabilities.py
# Goal:           Probabilidades de elección condicionales (CCPs)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from utils import dims, stack_states
from primitives import choice_probs
from bellman import choice_values, log_buy_probs


class CCP(NamedTuple):
    keep: jnp.ndarray    # (n,) sobre X; 0 en term y none
    purge: jnp.ndarray   # (n,) sobre X; en none = quedarse sin coche
    trade: jnp.ndarray   # (n,) sobre X; comprar algún coche
    buy: jnp.ndarray     # (n,) sobre H: Pr(comprar h | trade); 0 en la columna none
    repair: jnp.ndarray  # (n,) sobre H: Pr(reparar | h); 0 en nuevos y none


@partial(jax.jit, static_argnames="g")
def ccps(EV, P, g):
    J, A, S, n_act, n = dims(g)
    v = choice_values(EV, P, g)

    p_act = choice_probs([v.keep, v.purge_act, v.trade_act], g.sigma)
    p_term = choice_probs([v.purge_term, v.trade_term], g.sigma)
    p_none = choice_probs([v.stay_none, v.trade_none], g.sigma)

    zJ, z0 = jnp.zeros(J), jnp.zeros(())
    # Dentro del nido trade: la distribución de qué se compra NO depende de x
    # (separabilidad aditiva + GEV).  Nido de s dentro de (j, d): ver buy_inclusive.
    p_buy = jnp.exp(log_buy_probs(v.buy, g))

    # Etapa 2: solo depende de h (análogo al scrap endógeno de Gillingham, ec. 18)
    p_rep = choice_probs(v.repair, g.sigma_repair)[1]

    return CCP(
        keep=stack_states(p_act[0], zJ, z0),
        purge=stack_states(p_act[1], p_term[0], p_none[0]),
        trade=stack_states(p_act[2], p_term[1], p_none[1]),
        buy=jnp.concatenate([p_buy, jnp.zeros(1)]),
        repair=stack_states(p_rep, zJ, z0),
    )
