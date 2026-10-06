# `claude/scripts/modelo_fin/`: modelo, equilibrio y Monte Carlo

Reemplaza a `bellman_claude.py` y `params_claude.py`, que ahora están separados por
componentes, siguiendo el layout del repo de Rust (`scripts/main/`).

## Archivos

| archivo | qué tiene | depende de |
|---|---|---|
| `params.py` | `Params` (frozen, estático en jit) | — |
| `utils.py` | `dims`, grid de s, layout X/H (`split_states`, `stack_states`, `decode_states`), `initial_prices`, `gmres` (anidable) | params |
| `primitives.py` | utilidad, `sell_cost` (Ts), transición de s (`s_transition_rows`, `s_transition`), `emax`, `choice_probs` | utils |
| `bellman.py` | `continuation_values`, `repair_stage`, nido de compra (`buy_inclusive`, `log_buy_probs`), `choice_values`, `T`, `nk_step`, `solve_bellman` | primitives |
| `probabilities.py` | `CCP`, `ccps` | bellman |
| `transitions.py` | **sin matrices (v1.6):** `holding_kernel`, `holding_apply` (Q v), `holding_rapply` (w Q), `M_apply` (M v), `M_rapply` (q M), `holding_distribution` (q Ω), `stationary_q`. **Densas, solo pruebas:** `physical_matrices` (Q0, Q1), `holding_transition`, `trade_matrices` (Ω), `transition_matrix` (M), `stationary_distribution`. Más `observed_s_transition` | probabilities |
| `ED.py` | `market_components`, `excess_demand_log`, jacobiano sin matriz (`make_jvp`), `solve_equilibrium` | todo lo anterior |
| `gen_dataset.py` | regímenes de R (`solve_regimes`), `simulate_panel`, vistas (`view`, `to_cells`) | ED |
| `loglikelihood.py` | datos a celdas, verosimilitudes oráculo / ingenuo / hx, `estim_ll` | primitives |
| `montecarlo.py` | `MCDesign`, `prepare`, `one_mc`, `montecarlo` | gen_dataset, loglikelihood |
| `main.py` | CLI para correr local o por bloques en la supercomputadora | montecarlo |
| `tests.py` | pruebas (`python tests.py [--ed] [--n_s 12]`) | todo |

`nk_step` importa `ccps` y `M_apply` dentro de la función para evitar el ciclo
bellman -> transitions -> probabilities -> bellman.

## Sin matrices densas (v1.6)

Ningún paso del modelo arma matrices de n x n (salvo las pruebas a tamaño chico). Con
a_max = 25 y n_s = 100 (n = 7,204), cada matriz densa eran ~415 MB.

- **Q como kernel:** `holding_kernel` guarda (J, A-1, S, S) con la mezcla de la etapa 2
  ya integrada. Q v y w Q son einsums sobre ese kernel.
- **M = ΩQ por estructura:** Ω_keep es diagonal, Ω_trade = trade ⊗ buy es de rango 1 y
  Ω_purge es una columna:

      M v = keep ⊙ (Q v) + trade <buy, Q v> + purge v(none)

- **Distribución estacionaria exacta, sin resolver sistemas** (`stationary_q`). Todo
  coche activo viene de una compra, y las compras se reparten según `buy`, que no
  depende de x. Con masa compradora 1, q se arma con una recursión en la edad (cada
  edad sale de la anterior por el kernel); term y none salen de los flujos, y al final
  se normaliza. Solo suma productos no negativos: precisión relativa de máquina aun
  en celdas de 1e-15 (el solve denso solo daba precisión absoluta, con q de hasta
  −1e-16).
- **Newton-Kantorovich con GMRES** (`nk_step`): (I − βM) d = EV − T(EV) con productos
  M v. Converge en pocas iteraciones porque M = (keep Q, nilpotente: la edad solo sube)
  + una parte de rango 2.
- **`utils.gmres`:** GMRES propio con jnp y lax. El de `jax.scipy` no se puede anidar
  (el GMRES de P en `ED.py` llama dentro a uno de la Bellman), y daba
  NotImplementedError.
- **Simulación** (`gen_dataset._next_state`): accidente con prob s y luego s' ~ F_r, en
  lugar de muestrear filas de Q. Los regímenes guardan F en lugar de Q0 y Q1 (los .npz
  viejos hay que regenerarlos con `--solve-only`).

**Verificado (`tests.py`, `test_matrix_free`):**
- Q v, w Q, M v y q M contra las densas: ~1e-15.
- q por recursión contra el solve denso: 2e-15.
- Equilibrio completo contra la versión densa (tag `matrices-densas`): EV igual, P a
  1e-7 (dentro de la tolerancia de ED). Con n_s = 24 pasó de 124 s a 56 s.
- Bellman + q con a_max = 25 y n_s = 100: 9 s en laptop, sup|T(EV) − EV| = 1e-14,
  q = qM a 4e-17.
- Simulador: tasa de accidentes, E[s'] y fracción sin coche coinciden con el modelo
  dentro del error de muestreo.

## Cambios de modelo en esta versión

1. **Nido de s al comprar (`sigma_s`, default 0.1).** Antes cada celda (j, d, s) era
   una alternativa logit con su propio shock. El equilibrio dependía del tamaño del
   grid y los precios de celdas casi vacías explotaban (un coche de 1 año a 399 contra
   142 del nuevo). Ahora el árbol de compra es

       trade -> {(j, d) usados, nuevos j}   escala sigma_trade
       (j, d) -> s                          escala sigma_s <= sigma_trade

   Con sigma_s chico la elección de s es casi por valor neto: el precio es hedónico en
   s. El jacobiano βM sigue siendo exacto (es GEV).
2. **ED en logs con piso** (`ed_floor` = 1e-10). Por debajo de ~1e-13, q es ruido
   numérico del solve lineal. Con un piso de 1e-14 el jacobiano por diferencias finitas
   salía basura en esas celdas.

## Equilibrio (`ED.py`)

- **Ecuaciones:** D(h) = trade_mass · Pr(h | trade) y S(x) = q(x)(1 − Pr(keep | x)).
  Se resuelve ED_log = log D − log max(S, piso) = 0.
- **Solver:** tâtonnement con el jacobiano diagonal (-mu/sigma_s) hasta `tat_tol`, y
  luego Newton-Krylov. J·v se calcula con jvp; el efecto de P en EV entra por la
  función implícita, (I − βM) dEV = (dT/dP) dP. Se resuelve con GMRES y precondicionador
  diagonal, más búsqueda lineal.
- **Verificado:** J·v contra diferencias finitas, error relativo de 5e-5 (el error
  restante es redondeo de las diferencias finitas). Con n_s = 12 converge de forma
  cuadrática (1e-3 → 1e-7 → 3e-11).
- **Celdas sin masa:** su precio sale donde D = piso y no tiene significado económico.
  Se marcan en `Equilibrium.empty`.

## Monte Carlo

    python main.py --solve-only --T 13 --spread 0.3          # equilibrios -> output/regimes_*.npz
    python main.py --reps 0:10 --N 20000 --T 13 --spread 0.3  # bloque de réplicas -> output/mc_*.csv
    python main.py --smoke                                    # prueba chica

- **Estimadores** (`loglikelihood.py`):
  - **oráculo:** ve r;
  - **ingenuo:** una sola F, sin reparación;
  - **hx:** mezcla con p_t(h) libre por (año, j, d) más una función cuadrática en s
    (`flexible`), o un logit en R (`logit_R`).
  - s_repair = exp(·) > 0 fija el orden de los componentes (Assumption 3(iii)).
- **Métricas por réplica:** las estimaciones de (s_const, s_age, s_persist, s_repair,
  s_sigma) de cada estimador, y el sesgo y RMSE de Pr(repair | h) recuperada, total y
  por edad. Con `--se`, errores estándar por hessiano para medir cobertura.
- **Para la supercomputadora:** primero un job con `--solve-only` (los equilibrios son
  lo caro y se calculan una sola vez). Luego un array con bloques `--reps a:b`; cada
  bloque escribe su CSV.

## Pendientes

1. **Segunda etapa estructural** (mu, sigma_repair, costos). `g` es estático en jit,
   así que cada valor de parámetros recompila. Hay que pasar los parámetros que se
   estiman como un pytree dinámico separado de `g`.
2. **Escenario de celdas** (datos daneses): `to_cells` ya genera los datos; falta el
   estimador con s latente.
3. **Escenario km:** requiere manejo en el modelo (`idea_km.md`).
4. **Calibración de s.** Con los defaults, la probabilidad de accidente es de 3–20% al
   año (Gillingham, Tabla 4: 0.4–2%) y en equilibrio la gran mayoría de los hogares
   queda sin coche. Ver la sección "Diagnóstico" abajo.
5. **a_max = 7** con parámetros de Gillingham calibrados para 25 edades: los coches
   viven 6 años. Revisarlo junto con la calibración de s.

## Diagnóstico: equilibrio y calibración de s (n_s = 12–24, en laptop)

| calibración | sin coche | P por edad (j = 0, a = 1..6) | Pr(repair) por edad |
|---|---|---|---|
| defaults (accidentes de 3–20% al año) | 0.92–0.96 | 131, 110, 87, 67, 42, 22 | — |
| s tipo Gillingham, Tabla 4 (`s_new` = 0.004, `s_const` = 0.002, `s_age` = 0.0003, `s_sigma` = 0.003, `s_repair` = 0.004, `s_max` = 0.1) | 0.62 | 137, 112, 86, 64, 37, 15 | 0.17 → 0.05 |

Gillingham tiene cerca de 44% de observaciones sin coche (Tabla 3). La dinámica de s
inventada mata los coches muy rápido. Con tasas de accidente realistas el resultado se
acerca, y lo que falta probablemente es a_max = 7. **Los defaults no se cambiaron:** es
una decisión de calibración pendiente.

## Prueba de humo del Monte Carlo (`python main.py --smoke`)

Con n_s = 12, T = 2, N = 3,000 y 1 réplica, la cadena completa corre: equilibrios (2 ×
~3 min) → simulación → 3 estimadores → CSV.

| | s_repair | s_sigma | s_persist |
|---|---|---|---|
| verdad | 0.060 | 0.015 | 1.00 |
| oráculo | 0.064 | 0.0125 | 0.99 |
| ingenuo | 0 (impuesto) | 0.031 | 0.91 |
| hx | 0.066 | 0.013 | 1.02 |

Error de Pr(repair | h) en hx: sesgo de -0.012 y RMSE de 0.12. Con 150 celdas es solo una
verificación de que el código funciona, no un resultado.

## Calibración de R (de la versión anterior)

Con `sigma_repair` = 1, la base de R se multiplicó por 5, a (15, 12.5, 25) mil DKK, para
que la reparación "por puro ruido" sea de 5–15% y no de ~40%. Detalle en
`estructura_decision.md`.
