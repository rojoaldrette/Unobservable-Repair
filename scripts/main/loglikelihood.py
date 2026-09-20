# _____________________________________________________________________________
#
# Project:        Rust-replication
#
# Script:         scripts/loglikelihood.py
# Goal:           Funciones auxiliares del proyecto
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________


# Packages ____________________________________________________________________

import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from probabilities import calc_probs
from scipy.optimize import minimize
from utils import make_params

# The meat _____________________________________________________________________

def treat_data(cell_df):
    mileage = jnp.asarray(cell_df["mileage"].values)
    n = jnp.asarray(cell_df["frequency"].values)
    r = jnp.asarray(cell_df["replace"].values)

    return mileage, n, r




def log_lh(theta, mileage, n, r):

    p1, _ = calc_probs(theta)

    p = p1[mileage]
    eps = 1e-10
    p = jnp.clip(p, eps, 1 - eps)

    ll = jnp.sum(
        r * jnp.log(p)
        + (n - r) * jnp.log1p(-p)
    )

    return ll


@jax.jit
def objective(theta, mileage, n, r):

    theta1 = make_params(theta)

    return -log_lh(theta1, mileage, n, r)


def wrapper(theta0, data):

    mileage, n, r = treat_data(data)

    return objective(theta0, mileage, n, r)


def estim_ll(theta, data):

    diff = jax.grad(wrapper)
    
    start_time = time.perf_counter()

    result = minimize(
        fun=wrapper,
        x0=np.array([8.0, 0.09]),
        jac=diff,
        args=(data),
        method="L-BFGS-B",
        bounds=[
            (1e-6, None), 
            (1e-6, None) 
        ]
        )
    end_time = time.perf_counter()
    print(result)

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")

    return result






if __name__ == "__main__":

    df = pd.read_csv('output/dataset_cb_example.csv')
    theta = [1.0, 1.0]
    
    res = estim_ll(theta, df)







