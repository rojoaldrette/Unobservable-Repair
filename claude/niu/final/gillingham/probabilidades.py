# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/gillingham/probabilidades.py
# Goal:           CCPs, matrices de transición y distribución estacionaria (un tipo)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           07/10/2026
#
# _____________________________________________________________________________
'''
- CCPs (ecs. 12-18): keep, purge, trade sobre X; buy = Pr(h | trade) sobre H (no depende
  de x: el costo de transacción es una constante para cada x); scrap = Pr(chatarrear |
  se deshace de un coche activo) (ec. 18).
- Q (ecs. 27-28): física, H -> X.  used(j, d) -> act(j, d+1) con 1 - alpha(j, d) o
  term(j) con alpha(j, d); d = A-1 -> term(j) seguro; new(j) -> act(j, 1) o term(j);
  none -> none.
- M = Ω Q (ec. 29), con Ω[x, h] = keep(x) 1{h = x} + trade(x) buy(h) + purge(x) 1{h = none}.
  q = q M es la distribución estacionaria al inicio del periodo (Teorema 1).
- Resultados observables (lo que usa la verosimilitud), sobre X:
      0 keep | 1 purge, coche vendido | 2 purge, coche chatarreado
             | 3 trade, coche vendido | 4 trade, coche chatarreado
  En la terminal "chatarreado" es forzoso; sin coche, 1 = seguir sin coche y 3 = comprar.
'''

from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from utils import dims, stack_states, choice_probs
from bellman import choice_values, accident_prob

OUTCOMES = ("keep", "purge_vende", "purge_chatarra", "trade_vende", "trade_chatarra")


class CCP(NamedTuple):
    keep: jnp.ndarray     # (n,) sobre X
    purge: jnp.ndarray    # (n,) sobre X
    trade: jnp.ndarray    # (n,) sobre X
    buy: jnp.ndarray      # (n,) sobre H; 0 en none
    scrap: jnp.ndarray    # (n,) sobre X: Pr(chatarrear | se deshace); 1 en term, 0 en none


def ccps(EV, P, th, cfg):
    J, A, n_act, n = dims(cfg)
    v = choice_values(EV, P, th, cfg)
    p_act = choice_probs([v["keep"], v["purge_act"], v["trade_act"]], cfg.sigma)
    p_term = choice_probs([v["purge_term"], v["trade_term"]], cfg.sigma)
    p_none = choice_probs([v["stay_none"], v["trade_none"]], cfg.sigma)
    b = v["buy"] / cfg.sigma_trade
    scrap = choice_probs([v["scrap"], v["sell"]], th["sigma_sell"])[0]
    zJ, z0 = jnp.zeros(J), jnp.zeros(())
    return CCP(keep=stack_states(p_act[0], zJ, z0),
               purge=stack_states(p_act[1], p_term[0], p_none[0]),
               trade=stack_states(p_act[2], p_term[1], p_none[1]),
               buy=jnp.concatenate([jnp.exp(b - logsumexp(b)), jnp.zeros(1)]),
               scrap=stack_states(scrap, jnp.ones(J), z0))


def log_buy_probs(EV, P, th, cfg):
    # log Pr(comprar used(j, d) | trade), (J, A-1): exacto aun para probabilidades diminutas
    J, A, n_act, n = dims(cfg)
    b = choice_values(EV, P, th, cfg)["buy"] / cfg.sigma_trade
    return (b - logsumexp(b))[:n_act].reshape(J, A - 1)


def physical_matrix(th, cfg):
    # Q (n, n), H -> X
    J, A, n_act, n = dims(cfg)
    al = accident_prob(th, cfg)
    eyeJ = jnp.eye(J)
    surv = jnp.concatenate([1 - al[:, 1:A - 1], jnp.zeros((J, 1))], axis=1)          # (J, A-1)
    act_act = jnp.einsum("jk,de,jd->jdke", eyeJ, jnp.eye(A - 1, k=1), surv).reshape(n_act, n_act)
    act_term = ((1 - surv)[:, :, None] * eyeJ[:, None, :]).reshape(n_act, J)
    first = (jnp.arange(A - 1) == 0).astype(float)
    new_act = jnp.einsum("jk,e,j->jke", eyeJ, first, 1 - al[:, 0]).reshape(J, n_act)
    new_term = al[:, 0][:, None] * eyeJ
    return jnp.concatenate([
        jnp.concatenate([act_act, act_term, jnp.zeros((n_act, 1))], axis=1),
        jnp.concatenate([new_act, new_term, jnp.zeros((J, 1))], axis=1),
        jnp.concatenate([jnp.zeros((1, n - 1)), jnp.ones((1, 1))], axis=1)])


def transition_matrix(c, Q):
    # M = Ω Q (n, n), X -> X
    e_none = jnp.zeros(Q.shape[0]).at[-1].set(1.0)
    return c.keep[:, None] * Q + c.trade[:, None] * (c.buy @ Q)[None, :] + c.purge[:, None] * e_none


def stationary(M):
    # q = q M, sum(q) = 1  (solve denso: n es chico)
    n = M.shape[0]
    A_ = jnp.concatenate([(M.T - jnp.eye(n))[:-1], jnp.ones((1, n))])
    return jnp.linalg.solve(A_, jnp.zeros(n).at[-1].set(1.0))


def outcome_probs(c):
    # C (n, 5) sobre X: probabilidad de cada resultado observable
    return jnp.stack([c.keep, c.purge * (1 - c.scrap), c.purge * c.scrap,
                      c.trade * (1 - c.scrap), c.trade * c.scrap], axis=1)
