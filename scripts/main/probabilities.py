# _____________________________________________________________________________
#
# Project:        Rust-replication
#
# Script:         scripts/probabilities.py
# Goal:           Funciones auxiliares del proyecto
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________


# Packages ____________________________________________________________________


import jax.numpy as jnp
from bellman import make_v_0, make_v_1, solve_bellman
from utils import make_mileage_grid
from params import Params

# Main function ____________________________________________________________________

def calc_probs(g):
    
    V_final = solve_bellman(g)
    x = make_mileage_grid(g)
    # I checked it worked so I dropped v_0 to keep it normalized
    # If I see too much bias then I will add it
    v_1 = make_v_1(x, g, V_final)

    p1 = jnp.exp(v_1 - V_final)
    p0 = 1 - p1

    return p1, p0


if __name__ == "__main__":
    # Pruebas
    g = Params()
    p1, p0 = calc_probs(g)
    print(p1)
    print(p0)
    print(p1 + p0)

