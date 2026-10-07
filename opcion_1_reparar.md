# Opción 1 reparar

## Idea

Reparar cambia la probabilidad de descomponerse **en el periodo actual**, y el siguiente
periodo se parte del s ya reparado (con edad a+1).

    s_eff(r, s) = clip(s - s_repair * r, s_min, s_max)     # prob de descomponerse hoy
    s'          = s_const_j + s_age * a + s_persist * s_eff + eta   # si sobrevive

Con s_persist = 1 la media de s' es idéntica a la del código actual (opción 2), así que
F no cambia. Solo cambia la probabilidad de descomponerse, que ahora depende de r.

El código actual implementa la **opción 2**: reparar solo mueve la media de s'; la
probabilidad de descomponerse hoy es el s actual para r = 0 y r = 1.

## Cambios en bellman.py

Helper nuevo:

```python
def breakdown_prob(g):
    # (2, S): prob de descomponerse este periodo según r
    grid = make_s_grid(g)
    return jnp.stack([grid, jnp.clip(grid - g.s_repair, g.s_min, g.s_max)])
```

continuation_values:

```python
pb = breakdown_prob(g)[:, None, None, :]                       # (2, 1, 1, S)
survive = (1.0 - pb) * EV_next + pb * term[None, :, None, None]
```

physical_matrices:

```python
pb = breakdown_prob(g)
surv = (1.0 - pb)[:, None, None, :, None] * F[:, :, 1:]
p_term = jnp.where(is_last[None], 1.0, pb[:, None, :])          # (2, A-1, S)
# act_to_term depende de r: construirlo dentro de assemble(r) con p_term[r]
```

s_mean_next (solo si s_persist != 1):

```python
s_eff = jnp.clip(s - g.s_repair * r, g.s_min, g.s_max)
return c + g.s_age * d + g.s_persist * s_eff
```

El coche nuevo no cambia (nunca se repara).

## Implicaciones para identificación (Hu & Xin)

1. **Selección por supervivencia.** Solo se observa s_{t+1} de coches que sobreviven, y los
   reparados sobreviven más. Los pesos de la mezcla (ec. 6.4) ya no son las CCPs:

       Pr(r | s, sobrevive, conservó) ∝ p(r | s) * (1 - s_eff(r, s))

   Ajustar observed_kept_transition:

   ```python
   pb = breakdown_prob(g)[:, None, None, :]
   wk = pk * (1 - pb[0]); wr = pr * (1 - pb[1])
   w = wr / (wk + wr)
   ```

   Si no se corrige, el Monte Carlo confunde sesgo de selección con error del estimador.

2. **Momento adicional.** La tasa de descompostura de coches conservados también es mezcla en r:

       Pr(descompostura | s, conservó) = Σ_r p(r | s) * s_eff(r, s)

   Si los datos dicen qué coches conservados salen del parque, es una segunda ecuación sobre
   las mismas CCPs (ayuda a identificar o sirve para sobreidentificación).

## Pendiente por discutir

- ¿Cuándo se mide s en los datos? La opción 1 requiere que el s_t observado sea **previo** a
  la reparación (p. ej. en la inspección, antes de decidir). Si se mide después de reparar,
  el s observado ya incorpora r y el planteamiento cambia.
- Opción 1 vs opción 2 según lo que permitan los datos.
