# _____________________________________________________________________________
#
# Project:        Rust-replication
#
# Script:         scripts/params.py
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:

    # Structural params ###################

    beta: float = 0.97

    # Utility/cost of keeping the car
    mileage_cost: float = 0.013

    # Cost of replacing the car
    replacement_cost: float = 7.0


    # State variable #####################

    # Mileage grid
    mileage_min: float = 0.0
    mileage_max: float = 90.0
    mileage_step: float = 1.0

    # Probabilities of transition
    q1: float = 0.1 # +0 
    q2: float = 0.5 # +1
    q3: float = 0.4 # +2

    # Value function iteration #################################

    vfi_tol: float = 1e-8
    vfi_max_iter: int = 10_000

    # Monte Carlo ########################################

    mc_replic: int = 250