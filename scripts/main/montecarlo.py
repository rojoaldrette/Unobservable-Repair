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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gen_dataset import gen_dataset
from loglikelihood import estim_ll
from utils import make_params

# Notes
"""
Still missing some power by chunking and vmaping the montecarlo
"""



# Packages ____________________________________________________________________


def one_mc(theta, seed, theta0):
    g1 = make_params(theta)

    df = gen_dataset(g1, seed)

    res = estim_ll(theta0, df)

    return np.asarray(res.x), df


def montecarlo(R, theta):

    estimates = []
    theta0 = [1.0, 1.0]
    lista_df = []

    for r in range(R):
        
        theta_hat, df1 = one_mc(theta, r, theta0)
        lista_df.append(df1)
        estimates.append(theta_hat)

        print(
            f"Replication {r + 1}/{R}: "
            f"theta_hat = {theta_hat}"
        )

    df = pd.concat(lista_df)

    return np.asarray(estimates), df


def results_mc(estimates, theta_true, graph=True):

    bias = estimates.mean(axis=0) - theta_true

    std = estimates.std(axis=0, ddof=1)

    rmse = np.sqrt(
        np.mean((estimates - theta_true)**2, axis=0)
    )

    print("True parameters:")
    print(theta_true)

    print("\nMean estimates:")
    print(estimates.mean(axis=0))

    print("\nBias:")
    print(bias)

    print("\nStd. deviation:")
    print(std)

    print("\nRMSE:")
    print(rmse)




def graph_results(estimates, theta_true):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Replacement cost
    axes[0].hist(estimates[:, 0], bins=30)
    axes[0].axvline(
        theta_true[0],
        linestyle="--",
        linewidth=2
    )
    axes[0].set_title("Replacement cost")
    axes[0].set_xlabel(r"$\hat{C}$")
    axes[0].set_ylabel("Frequency")

    # Mileage cost
    axes[1].hist(estimates[:, 1], bins=30)
    axes[1].axvline(
        theta_true[1],
        linestyle="--",
        linewidth=2
    )
    axes[1].set_title("Mileage cost")
    axes[1].set_xlabel(r"$\hat{c}_m$")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Second graph

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(estimates[:, 0])
    axes[0].axhline(
        theta_true[0],
        linestyle="--",
        linewidth=2
    )
    axes[0].set_title("Replacement cost estimates")
    axes[0].set_xlabel("Monte Carlo replication")
    axes[0].set_ylabel(r"$\hat{C}$")

    axes[1].plot(estimates[:, 1])
    axes[1].axhline(
        theta_true[1],
        linestyle="--",
        linewidth=2
    )
    axes[1].set_title("Mileage cost estimates")
    axes[1].set_xlabel("Monte Carlo replication")
    axes[1].set_ylabel(r"$\hat{c}_m$")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Pruebas
    theta=[8.0, 0.015]
    theta0 = [1.0, 1.0]

    results, df_final = montecarlo(1000, theta)

    results_mc(results, theta)

    graph_results(results, theta)





