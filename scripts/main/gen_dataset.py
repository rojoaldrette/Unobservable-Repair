# _____________________________________________________________________________
#
# Project:        Rust-replication
#
# Script:         scripts/gen_dataset.py
# Goal:           Funciones auxiliares del proyecto
#
# Author:         Rodrigo Antonio Aldrette Salas
# Mail:           raaldrettes@colmex.mx
#
# Date:           28/08/2026
#
# _____________________________________________________________________________


# Packages ____________________________________________________________________

import jax
import pandas as pd
import jax.numpy as jnp
from params import Params
from probabilities import calc_probs


T = 120 # Time periods
M = 50 # Buses

def sim_buses(p1, g, T=120, n_bus=50, seed=123):
    key = jax.random.PRNGKey(seed)

    mileage = jnp.zeros((n_bus, T), dtype=int)
    replace = jnp.zeros((n_bus, T), dtype=int)
        
    for t in range(T):
        m = mileage[:, t]
        prob = p1[m]

        key, subkey = jax.random.split(key)

        draw = jax.random.uniform(
            subkey,
            shape=(n_bus,)
        )

        d = (draw < prob).astype(int)
        replace = replace.at[:, t].set(d)

        key, subkey = jax.random.split(key)

        # Mileage increment: 0, 1, or 2
        dx = jax.random.choice(
            subkey,
            jnp.array([0, 1, 2]),
            shape=(n_bus,),
            p=jnp.array([g.q1, g.q2, g.q3])
        )

        # If replaced -> new bus starts at dx
        # If kept -> mileage increases
        m_next = jnp.where(
            d == 1,
            dx,
            m + dx
        )

        # Prevent going beyond p1's state space
        m_next = jnp.minimum(
            m_next,
            len(p1) - 1
        )

        if t < T - 1:
            mileage = mileage.at[:, t + 1].set(m_next)

    return mileage, replace



def gen_dataset(g, seed, T=120, n_bus=50, cell_based=True):

    p1, _ = calc_probs(g)

    mil, rep = sim_buses(p1, g, seed=seed)

    df = pd.DataFrame({
    "bus": jnp.repeat(
        jnp.arange(n_bus),
        T
    ),
    "t": jnp.tile(
        jnp.arange(T),
        n_bus
    ),
    "mileage": mil.flatten(),
    "replace": rep.flatten()
    })

    if cell_based == True:
        cell_df = (df.groupby('mileage')
                    .agg(frequency= ('replace', 'size'),
                         replace=('replace', 'sum'))
                    .reset_index()
                    )
        cell_df['keep'] = (cell_df['frequency'] - cell_df['replace'])

        df = cell_df.copy()

    return df



if __name__ == "__main__":
    # Pruebas
    g = Params()
    df = gen_dataset(g, 123, cell_based=False)
    df.to_csv('output/dataset_example.csv', index=False)

    df2 = gen_dataset(g, 123, cell_based=True)
    df2.to_csv('output/dataset_cb_example.csv', index=False)




