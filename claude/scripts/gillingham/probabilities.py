# _____________________________________________________________________________
#
# Project:        Rust-replication  ->  Autos (Gillingham et al.) + reparaciones
#
# Script:         claude/scripts/gillingham/probabilities.py
# Goal:           CCPs (ecs. 12–18 de Gillingham)
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
from jax.scipy.special import logsumexp

from utils import dims, stack_states
from bellman import choice_values, choice_probs


class CCP(NamedTuple):
    keep: jnp.ndarray    # (n,) sobre X
    purge: jnp.ndarray   # (n,) sobre X
    trade: jnp.ndarray   # (n,) sobre X
    buy: jnp.ndarray     # (n,) sobre H, Pr(h | trade); 0 en none
    scrap: jnp.ndarray   # (J, A-1): Pr(chatarrear | se deshace del coche activo), ec. 18


@partial(jax.jit, static_argnames="g")
def ccps(EV, P, g):
    J, A, n_act, n = dims(g)
    v = choice_values(EV, P, g)
    p_act = choice_probs([v.keep, v.purge_act, v.trade_act], g.sigma)
    p_term = choice_probs([v.purge_term, v.trade_term], g.sigma)
    p_none = choice_probs([v.stay_none, v.trade_none], g.sigma)
    p_buy = jnp.exp(v.buy / g.sigma_trade - logsumexp(v.buy / g.sigma_trade))
    zJ, z0 = jnp.zeros(J), jnp.zeros(())
    return CCP(
        keep=stack_states(p_act[0], zJ, z0),
        purge=stack_states(p_act[1], p_term[0], p_none[0]),
        trade=stack_states(p_act[2], p_term[1], p_none[1]),
        buy=jnp.concatenate([p_buy, jnp.zeros(1)]),
        scrap=choice_probs([v.scrap, v.sell], g.sigma_sell)[0],
    )
