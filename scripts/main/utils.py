# _____________________________________________________________________________
#
# Project:        Rust-replication
#
# Script:         scripts/utils.py
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________

import jax.numpy as jnp
from params import Params
from dataclasses import replace

def make_mileage_grid(p):
    return jnp.arange(
        p.mileage_min,
        p.mileage_max + p.mileage_step,
        p.mileage_step
    )

def make_params(theta):
    g = Params()
    g1 = replace(g, replacement_cost=theta[0], mileage_cost=theta[1])
    return g1
