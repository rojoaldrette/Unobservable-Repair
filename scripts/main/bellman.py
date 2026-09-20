# _____________________________________________________________________________
#
# Project:        Rust-replication
#
# Script:         scripts/bellman.py
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________


# Packages and config ____________________________________________________________________

import jax.numpy as jnp
from jaxopt import FixedPointIteration
from params import Params
from utils import make_mileage_grid

'''
Quick comments:

1. 'g' is the parameter that is controlled from main, it is called like 
g = Params(), can be used for internal debugging under if name = main

'''



# util functions ____________________________________________________________________

def make_u_1(m, g):
    u = -1 * (g.replacement_cost + g.mileage_cost * m)
    return u


def make_u_0(m, g):
    u = -1 * (g.mileage_cost * m)
    return u


def make_v_1(m, g, V):
    v = make_u_1(m, g) + g.beta * (g.q1*V[0] + g.q2*V[1] + g.q3*V[2])
    return v


def make_v_0(m, g, V):

    idx = jnp.arange(len(V))

    idx_1 = jnp.minimum(idx + 1, len(V) - 1)
    idx_2 = jnp.minimum(idx + 2, len(V) - 1)

    EV = (
        g.q1 * V[idx]
        + g.q2 * V[idx_1]
        + g.q3 * V[idx_2]
    )

    return make_u_0(m, g) + g.beta * EV


# Wrappers  ______________________________________________________________________

# Contraction mapping from bellman for V
def T(V, g):
    x = make_mileage_grid(g)
    v_1 = make_v_1(x, g, V)
    v_0 = make_v_0(x, g, V)
    return jnp.logaddexp(v_1, v_0)


def solve_bellman(g):
    # State variable grid
    x = make_mileage_grid(g)
    # Value function
    V_init = jnp.ones(len(x))

    T_fixed = lambda V: T(V, g)
    fpi = FixedPointIteration(fixed_point_fun=T_fixed, implicit_diff=False,
                                maxiter=g.vfi_max_iter, tol=g.vfi_tol)

    fp = fpi.run(V_init).params

    return fp
    

if __name__ == "__main__":
    #Pruebas
    g = Params()
    V_final = solve_bellman(g)
    print(V_final)


