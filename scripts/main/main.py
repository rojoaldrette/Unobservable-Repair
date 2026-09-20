# _____________________________________________________________________________
#
# Project:        Rust-replication
#
# Script:         scripts/main.py
# Goal:           Funciones auxiliares del proyecto
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________


# Packages ____________________________________________________________________

from probabilities import calc_probs
from gen_dataset import gen_dataset
from bellman import solve_bellman
from utils import make_params


# Main functioon ____________________________________________________________________


# Parameters ###################################

theta = [4.0, 0.09]
g = make_params(theta)
#g1 = replace(g, replacement_cost=4)


# Get probabilities ##############################
p1, p0 = calc_probs(g)


# Generate dataset #############################

df = gen_dataset(g, 123)


# Estimation ################################




