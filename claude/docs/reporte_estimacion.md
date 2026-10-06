# Reporte: estimación estructural, salidas y análisis (v1.7)

Qué se construyó, dónde se corre cada cosa, qué escribe y qué se verificó. **No se corrió
ninguna estimación real ni Monte Carlo**: solo pruebas de tamaño de juguete (a_max = 7,
n_s = 8, 3,000 hogares) para comprobar que el código funciona. Las corridas de verdad van
en la supercomputadora (`scripts/slurm_estimar.sh`).

---

## 1. Resumen

| pedido | dónde quedó |
|---|---|
| 1) verosimilitud con r observada (D0) | `modelo_fin/estructural.py`, `info = "oraculo"` |
| 2) verosimilitud con r no observada (D1) | `modelo_fin/estructural.py`, `info = "hx"` |
| 3) verosimilitud de Gillingham | `gillingham/loglikelihood.py` (ya existía); nuevo corredor `gillingham/estimar.py`, sobre sus datos o sobre el panel de modelo_fin |
| salidas en CSV | `claude/output/estimaciones/<modelo>/<tag>/` |
| gráficas y tablas | `claude/scripts/analisis/` -> `claude/output/analisis/<nombre>/` |
| dónde correr todo | `claude/scripts/slurm_estimar.sh` (pasos mf, mf_mc, gill, cruce, analisis) |

De paso quedó hecha buena parte de la Fase 2 del plan: parámetros dinámicos
(`theta.py`), dos tipos de hogar y Newton conjunto sobre (EV, P) con Krylov
(`equilibrium.py`). Falta de la Fase 2 el chatarreo endógeno opcional.

---

## 2. Cómo se corre

Todo desde la carpeta del script, con `PYTHONIOENCODING=utf-8` y `python -u`.

### En la supercomputadora (recomendado)

```
cd claude/scripts
sbatch --export=ALL,STEP=mf slurm_estimar.sh              # 1. verdad + réplica 0 de modelo_fin (D0 y D1) + panel
sbatch --array=1-9 --export=ALL,STEP=mf_mc slurm_estimar.sh   # 2. Monte Carlo, réplicas 10-99 (opcional)
sbatch --export=ALL,STEP=gill slurm_estimar.sh            # 3. Gillingham sobre sus datos
sbatch --export=ALL,STEP=cruce slurm_estimar.sh           # 4. Gillingham sobre el panel de modelo_fin (necesita 1)
sbatch --export=ALL,STEP=analisis slurm_estimar.sh        # 5. gráficas y tablas (necesita 1, 3, 4)
```

El diseño (a_max, n_s, regímenes, hogares, tipos) se fija arriba del script. Antes de
usarlo hay que ajustar la partición, los módulos y el entorno (`jax[cuda12]`).

### A mano

```
cd claude/scripts/modelo_fin
python -u estimar.py --reps 0:1 --guardar_panel -v          # una estimación (D0 y D1)
python -u estimar.py --reps 0:10                            # bloque de Monte Carlo
python -u estimar.py --smoke                                # juguete (minutos)
python -u tests.py --n_s 12 --estructural                   # pruebas, incluidas las nuevas

cd ../gillingham
python -u estimar.py --reps 0:1 --a_max 7 -v                                     # sus datos
python -u estimar.py --panel ../../output/estimaciones/modelo_fin/<tag>/panel_rep0.csv.gz --a_max 7 -v   # cruce

cd ../analisis
python main.py --mf ../../output/estimaciones/modelo_fin/<tag> \
               --gill ../../output/estimaciones/gillingham/<tag_cruce> \
               --gill ../../output/estimaciones/gillingham/<tag_propios> --nombre v1 --todas
```

### Opciones principales de `modelo_fin/estimar.py`

| opción | default | qué es |
|---|---|---|
| `--calib` | tesis_v0 | `defaults`, `gill_s` (s calibrada a la Tabla 4) o `tesis_v0` (gill_s + R realista + sigma_repair 0.3) |
| `--a_max`, `--n_s` | 7, 100 | tamaño del modelo |
| `--types`, `--f` | pareja pobre + soltero pobre, mitad y mitad | tipos de hogar |
| `--T`, `--spread` | 13, 0.3 | años-régimen de R: R_t = R exp(zeta_t), zeta en [−0.3, 0.3] |
| `--N`, `--K` | 20,000, 2 | hogares por régimen y años por hogar |
| `--infos` | oraculo,hx | qué estimadores correr |
| `--fix` | (nada) | parámetros fijos en la verdad |
| `--n_starts`, `--perturb` | 2, 0.1 | arranques perturbados además de la verdad |
| `--method` | krylov | `krylov` (memoria O(n), para tamaño completo) o `dense` (solo chico) |
| `--lbfgs_iter`, `--bhhh_iter` | 500, 50 | topes del optimizador |
| `--guardar_panel` | no | guarda el panel simulado (lo necesita el cruce con Gillingham) |
| `--rep_reporte` | la primera | réplica de la que se guardan precios, distribución y CCPs |

---

## 3. Los estimadores

Detalle matemático en `verosimilitud_estructural.md`. Todos son DNFXP: en cada
evaluación se resuelve el equilibrio (P no se observa) y el gradiente sale por la
función implícita.

### D0, r observada (`oraculo`)

Por hogar-año (τ, t, x, o, h, r, x'):

    log Pr = log Pr(o, h | x) + log Pr(r | h) + log F_r(s' | h) + log(1 − s(h))    si el coche sobrevive
           = log Pr(o, h | x) + log Pr(r | h) + log s(h)                            si muere

Pr(r | h) es la CCP de la etapa 2 del modelo: la decisión de reparar aporta directo.

### D1, r no observada (`hx`)

    log Pr = log Pr(o, h | x) + log[(1 − p(h)) F_0(s' | h) + p(h) F_1(s' | h)] + log(1 − s(h))

La mezcla de Hu & Xin, con p(h) = la CCP de reparar del modelo (no libre). La
variación de R_t entre años es la variable excluida.

### Gillingham (`gill_parcial`, `gill_completa`)

La réplica del paper (apéndice D): estado (j, a), sin s ni reparación. `gill_parcial` no
ve precios ni accidentes (un coche que sale puede ser accidente o chatarreo);
`gill_completa` ve el estado.

### El cruce: Gillingham sobre datos con reparación

`gillingham/estimar.py --panel` toma el panel de modelo_fin, borra s y r, pasa los
estados al layout (j, a) y junta los años (Gillingham no ve R). Los parámetros
"verdaderos" de utilidad y costos son los mismos en los dos modelos (Tablas 7-10), así
que la columna de Gillingham sobre datos con reparación **mide el sesgo de estimar
ignorando la reparación**. En las salidas aparece como `gill_parcial|modelo_fin`.

Advertencias del cruce:
- modelo_fin todavía no tiene chatarreo endógeno (Fase 2). Gillingham va a estimar
  chatarreo cerca de cero y atribuirá todas las salidas a accidentes.
- La prob. de accidente de Gillingham es un logit en la edad; en modelo_fin es s, que
  depende de la historia de reparación. Los parámetros de accidente no tienen
  contraparte directa.
- El RMSE de P no se calcula en el cruce: los espacios de estados son distintos. Para
  comparar precios está la gráfica `precios_edad` (P de modelo_fin promediado en s).

---

## 4. Salidas (CSV)

`claude/output/estimaciones/modelo_fin/<tag>/`:

| archivo | filas | columnas |
|---|---|---|
| `parametros_reps<a>-<b>.csv` | réplica × estimador × arranque × parámetro | rep, estimador, arranque, mejor, parametro, verdad, estimado, se, z, z_vs_verdad, ic95_lo, ic95_hi, ll, convergio |
| `resumen_reps<a>-<b>.csv` | réplica × estimador | ll, n_obs, n_celdas, convergio, iters, segundos, arranques, mismo_optimo, cond_B, P_rmse, y {sin_coche, tasa_reparacion, tasa_accidentes, edad_media, precio_medio} _verdad y _est |
| `precios.csv` | fuente × régimen × (j, a, s) | fuente, regimen, zeta, j, a, s_idx, s, P, q, sin_masa |
| `distribucion.csv` | fuente × tipo × estado (régimen central) | fuente, regimen, tipo, estado, j, a, s_idx, s, q |
| `ccps.csv` | fuente × tipo × (j, a, s) (régimen central) | fuente, regimen, tipo, j, a, s_idx, s, keep, purge, trade, repair, q |
| `config.json` | | argumentos, Params completos, tipos, zetas, parámetros libres |
| `verdad.npz` | | equilibrios verdaderos (z por régimen); se reutiliza si el diseño no cambia |
| `panel_rep<k>.csv.gz` | | el panel simulado (con `--guardar_panel`; no se sube a git) |

`fuente` ∈ {verdad, oraculo, hx}. q = fracción de hogares (agregada sobre tipos con f en
`precios`, por tipo en `distribucion` y `ccps`). `sin_masa`: celda sin oferta; su precio
no significa nada.

`claude/output/estimaciones/gillingham/<tag>/`: lo mismo sin s (j, a), con `scrap` en
`ccps.csv` y las estadísticas de Gillingham (chatarreo endógeno, accidentes, fracción
voluntaria) en `resumen`.

Ejemplo con pandas:

```python
import pandas as pd, glob
par = pd.concat(map(pd.read_csv, glob.glob(".../parametros_reps*.csv")))
par[par.mejor & (par.parametro == "s_repair")].groupby("estimador")[["estimado", "se"]].describe()
```

---

## 5. Análisis (`claude/scripts/analisis/`)

| archivo | qué hace |
|---|---|
| `datos.py` | lee una corrida (junta los bloques de réplicas), renombra las fuentes de Gillingham (`verdad_gill`, `…|modelo_fin`), agrega sobre tipos, promedia sobre s ponderando por q |
| `graficas.py` | las gráficas (abajo) |
| `tablas.py` | las tablas (abajo) |
| `main.py` | corre todo |

**Gráficas** (`graficas/`, PNG 200 dpi y PDF):

| archivo | qué muestra |
|---|---|
| `distribucion_edad` | q por edad y marca, todas las fuentes (modelo_fin y Gillingham) |
| `sin_coche` | fracción de hogares sin coche por fuente |
| `distribucion_s` | mapa de calor q(a, s) por marca y fuente |
| `precios_3d_marca<j>` | superficie P(a, s) por fuente; celdas sin masa en blanco; eje s recortado a donde hay coches |
| `precios_edad` | P(a) = media sobre s ponderada por q, con las P(a) de Gillingham encima |
| `ccps_edad` | Pr(reparar) y Pr(keep) por edad (ponderadas por q y tipo) |
| `mc_sesgo` | con varias réplicas: cajas de estimado − verdad por parámetro y estimador |

Colores fijos por fuente: verdad negro, D0 azul, D1 naranja, Gillingham sobre sus datos
aqua/violeta, Gillingham sobre datos con reparación rosa/amarillo.

**Tablas** (`tablas/`, cada una en CSV, LaTeX booktabs y Markdown):

| archivo | qué tiene |
|---|---|
| `tabla_parametros_principal` | tipo regresión: estimado con estrellas (H0: = 0) y (se) abajo, una columna por estimador, verdad al lado; pie con observaciones, LL y convergencia. Parámetros: mu, Tb, Tb sin coche, Ts, Ts inspección, sigma_repair, u_s, s_repair, s_sigma, sigma_sell |
| `tabla_sesgo_principal` | estimado − verdad con [z = (est − verdad)/se]: la prueba que importa en una simulación |
| `…_completa` (con `--todas`) | lo mismo con todos los parámetros (u0, u1 por marca y tipo, dinámica de s, accidentes) |
| `tabla_mercado` | sin coche, tasa de reparación, accidentes, edad media, precio medio, chatarreo, RMSE de P: verdad contra θ̂ |
| `tabla_mc` (+ `_numerica.csv`) | con varias réplicas: verdad, media, sesgo, RMSE, se medio, sd del MC, cobertura 95% |

---

## 6. Archivos nuevos y cambios

**Nuevos:**
- `modelo_fin/theta.py`: `Model`, `Economy` (pytrees), vector libre <-> θ.
- `modelo_fin/equilibrium.py`: equilibrio con tipos, Newton conjunto (dense o krylov),
  `solve_linear` para el gradiente implícito, `equilibrium_objects`.
- `modelo_fin/estructural.py`: datos a celdas, `cell_logp` (D0 y D1), `score_parts`
  (LL, gradiente, BHHH), `LLEval`, L-BFGS + BHHH.
- `modelo_fin/estimar.py`: corredor con salidas.
- `gillingham/estimar.py`: corredor con salidas y el cruce.
- `analisis/` (4 archivos), `slurm_estimar.sh`.

**Cambios en lo que existía** (sin cambiar resultados; las pruebas dan lo mismo):
- `primitives.sell_cost` usa jnp.where (tc_sell puede ser dinámico).
- `primitives.s_transition_rows`: bordes del grid finitos y lejanos en lugar de ±∞. Con
  ±∞ la derivada respecto a s_sigma daba NaN (0 · ∞). Los valores no cambian.
- `bellman.T_raw` y `probabilities.ccps_raw`: versiones sin jit; `T` y `ccps` siguen
  igual.
- `params.PAPER_TYPES` y `Types` (copia de los de Gillingham).
- `gen_dataset`: el panel trae los índices x, h, x_next; `simulate_economy` (varios tipos).
- `gillingham/loglikelihood.estim_bhhh`: pseudoinversa en lugar de inversa (igual si B
  es invertible; antes se caía con B singular).
- `tests.py --estructural`: pruebas nuevas.

---

## 7. Verificación (solo tamaño de juguete)

| prueba | resultado |
|---|---|
| equilibrio con Economy (1 tipo) contra `ED.solve_equilibrium` | P igual a 1e-13 |
| 2 tipos: krylov contra dense | z igual a 1e-13 |
| gradiente implícito contra diferencias finitas (29 parámetros, D0 y D1) | error relativo 2e-6 |
| pruebas anteriores (`tests.py`) | sin cambios |
| `estimar.py --smoke` (modelo_fin), `gillingham/estimar.py --smoke`, cruce, análisis | corren y escriben todo |

**Las estimaciones de juguete no dicen nada sobre el estimador.** Con 3,000 hogares, 2
regímenes y 6 edades, muchos parámetros no se identifican (cond(B) ~ 1e19) y el
optimizador se cortó a propósito. Sirven solo para probar el código.

---

## 8. Lo que agregué por prudencia

1. **Se guardan todos los arranques** (columna `arranque`, `mejor`). Era el pendiente 4:
   distinguir máximos locales de arranques mal pulidos.
2. **Un archivo por bloque de réplicas**, para que las tareas de un array de SLURM no
   escriban sobre el mismo CSV.
3. **`cond_B` en el resumen**: número de condición de la matriz BHHH. Si es enorme,
   alguna dirección no está identificada y sus se no son confiables (salen NaN).
4. **Pseudoinversa en BHHH**: con B casi singular ya no se cae la corrida.
5. **`--method dense | krylov`**: dense es exacto y sirve para probar; krylov es el que
   cabe a tamaño completo.
6. **`verdad.npz` se reutiliza** si el diseño no cambió (no se recalcula en cada bloque).
7. **`tasa_reparacion` y `edad_media` en θ̂ contra la verdad**: el "contrafactual" mínimo
   para ver si un sesgo en parámetros importa para lo que se reporta.

---

## 9. Pendientes y advertencias antes de correr en serio

1. **Calibración (Fase 3).** El default es `tesis_v0` con a_max = 7 y s en niveles. Lo
   acordado es log-odds y a_max = 25; hay que hacerlo antes de las corridas finales.
   El código de estimación no cambia: la transición de s está encapsulada en
   `primitives.s_transition`.
2. **sigma_repair = 0.3 es provisional.** Revisar que Pr(repair) quede en un rango con
   información una vez fija la calibración.
3. **Chatarreo endógeno opcional** (resto de la Fase 2). Sin él, el cruce con Gillingham
   compara modelos que difieren en dos cosas: reparación y chatarreo.
4. **Costo de cómputo desconocido a tamaño completo.** Cada evaluación resuelve T = 13
   equilibrios con 2 tipos (~21,600 incógnitas cada uno) y el gradiente pide ~30
   sistemas lineales por régimen. Conviene una corrida de prueba con `--T 3 --N 5000`
   en la GPU para medir tiempos antes del Monte Carlo.
5. **Gillingham sobre sus datos usa K = 10 años** (como el paper); modelo_fin usa K = 2
   por régimen × 13 regímenes. Las observaciones no son comparables uno a uno.
6. **El estado inicial de cada régimen sale de q_t** (estado estacionario): es la lectura
   de "años" como equilibrios separados (pendiente 10 de v1.0).
