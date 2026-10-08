# _____________________________________________________________________________
#
# Project:        Autos (Gillingham et al.) + reparaciones no observadas
#
# Script:         claude/niu/gillingham/bellman.py
# Goal:           Primitivas, valores por elección y operador de Bellman (un tipo)
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           07/10/2026
#
# _____________________________________________________________________________
'''
Gillingham et al., sec. 3.3, ecs. (2)-(11).  Todo es para UN tipo de hogar: `th` es el
dict de theta con los campos por tipo ya rebanados (utils.type_theta o jax.vmap).

Árbol de decisión (figura 1, con los nidos de marca y edad colapsados):

    sigma:        purge (sin coche) | keep | trade
    sigma_trade:  dentro de trade, qué h comprar: usados (j, d), d = 1..A-1, y nuevos j
    sigma_sell:   al deshacerse de un coche activo, vender (mu P - Ts) o chatarrear (mu p_scrap)

Vender/chatarrear es una decisión estática (ec. 18): misma probabilidad dentro de purge y
de trade, así que entra como un solo valor inclusivo `disposal`.

Tiempo: se comercia al inicio del periodo, se maneja la tenencia h durante el periodo y al
final h envejece (d -> d+1) o tiene un accidente (-> terminal) con prob alpha(j, d).
Un coche de edad A-1 llega a la terminal con certeza.  En la terminal el coche se
chatarrea a la fuerza (el dueño cobra p_scrap y elige como si no tuviera coche).
'''

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from utils import dims, split_states, stack_states, emax


# Primitivas ______________________________________________________________

def flow_utility(th, cfg):
    # u(j, d) = u0_j + u1_j d + u2_j d^2,  d = 0..A-1.  (J, A)
    d = jnp.arange(cfg.a_max)[None, :]
    return th["u0"][:, None] + th["u1"][:, None] * d + th["u2"][:, None] * d ** 2


def accident_prob(th, cfg):
    # alpha(j, d) = logit^-1(acc_int_j + acc_age_j d),  d = 0..A-1  (Tabla 4).  (J, A)
    d = jnp.arange(cfg.a_max)[None, :]
    return jax.nn.sigmoid(th["acc_int"][:, None] + th["acc_age"][:, None] * d)


def inspection_years(cfg):
    # a = 1..A-1: año de inspección (edad par >= inspect_age_min).  numpy: es estático
    a = np.arange(1, cfg.a_max)
    return (a % 2 == 0) & (a >= cfg.inspect_age_min)


def to_utils(cost, th, cfg):
    # Costos de transacción en utils (lectura "utils") o en miles de DKK (lectura "dkk")
    return cost * th["mu"] if cfg.tc_units == "dkk" else cost


def sell_cost(th, cfg):
    # Ts(a) en utils, a = 1..A-1
    ts = jnp.where(jnp.asarray(inspection_years(cfg)), th["tc_sell_inspect"], th["tc_sell"])
    return to_utils(ts, th, cfg)


def buy_costs(th, cfg):
    # (Tb de quien tiene coche o terminal, Tb de quien no tiene coche), en utils
    tb = to_utils(th["tc_buy"], th, cfg)
    tb_nc = to_utils(th["tc_buy_nocar"], th, cfg)
    return tb, (tb + tb_nc if cfg.nocar == "suma" else tb_nc)


# Valores ______________________________________________________________

def continuation_values(EV, th, cfg):
    # beta E[EV(x') | h]:  cv_used (J, A-1) para d = 1..A-1;  cv_new (J,)
    J, A, n_act, n = dims(cfg)
    act, term, none = split_states(EV, cfg)
    al = accident_prob(th, cfg)
    surv = (1 - al[:, 1:A - 1]) * act[:, 1:] + al[:, 1:A - 1] * term[:, None]   # d = 1..A-2
    cv_used = cfg.beta * jnp.concatenate([surv, term[:, None]], axis=1)       # d = A-1: terminal
    cv_new = cfg.beta * ((1 - al[:, 0]) * act[:, 0] + al[:, 0] * term)
    return cv_used, cv_new


def choice_values(EV, P, th, cfg):
    # P: precios de usados (J, A-1), miles de DKK.  Devuelve un dict de valores por elección.
    u = flow_utility(th, cfg)
    cv_used, cv_new = continuation_values(EV, th, cfg)
    _, _, none = split_states(EV, cfg)
    mu = th["mu"]
    tb, tb_none = buy_costs(th, cfg)

    sell = mu * P - sell_cost(th, cfg)[None, :]
    scrap = jnp.broadcast_to(mu * th["p_scrap"][:, None], P.shape)
    disposal = emax([scrap, sell], th["sigma_sell"])

    # Comprar h (sin el costo de transacción, que depende de quién compra)
    buy_used = u[:, 1:] - mu * P + cv_used
    buy_new = u[:, 0] - mu * th["p_new"] + cv_new
    buy = jnp.concatenate([jnp.ravel(buy_used), buy_new])
    iv = cfg.sigma_trade * logsumexp(buy / cfg.sigma_trade)

    no_car = cfg.beta * none                     # u(sin coche) = 0 (normalización)
    term_scrap = mu * th["p_scrap"]
    return dict(
        keep=u[:, 1:] + cv_used,
        purge_act=disposal + no_car, trade_act=disposal - tb + iv,
        purge_term=term_scrap + no_car, trade_term=term_scrap - tb + iv,
        stay_none=no_car, trade_none=-tb_none + iv,
        buy=buy, sell=sell, scrap=scrap,
    )


def T(EV, P, th, cfg):
    # Operador de Bellman Γ (ec. 9) sobre EV(x), x en X
    v = choice_values(EV, P, th, cfg)
    return stack_states(emax([v["keep"], v["purge_act"], v["trade_act"]], cfg.sigma),
                        emax([v["purge_term"], v["trade_term"]], cfg.sigma),
                        emax([v["stay_none"], v["trade_none"]], cfg.sigma))


def successive_approx(EV, P, th, cfg, n_iter):
    # n_iter aplicaciones de T (contracción de módulo beta)
    return jax.lax.fori_loop(0, n_iter, lambda i, V: T(V, P, th, cfg), EV)
