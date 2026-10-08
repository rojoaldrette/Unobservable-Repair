# Código de `modelo_tesis` y resultados del modelo teórico

La especificación vigente está en `modelo_fin.md`. Este doc dice cómo está hecho el código,
cómo se corre y qué sale. El plan anterior (2026-10-07, grid de 13 puntos) quedó
reemplazado.

Estado (2026-10-08): **modelo teórico completo hasta `gen_dataset`; núcleo de la
verosimilitud y script de tiempos listos.** Falta el optimizador y el Monte Carlo.

**Equilibrio en la estimación (`equilibrio.factor`, `solve_chord`):**

- **Inversa reutilizada:** la inversa del jacobiano se calcula una vez y se reutiliza. Para
  un θ cercano se dan pasos "de cuerda" que arrancan del predictor z + (dz/dθ)Δθ. Cada paso
  es una evaluación de F más un producto matriz-vector.
- **Refactorización:** solo cuando la cuerda se estanca (||F|| no baja a 0.7 de la anterior).
  Si aun así no converge, Newton completo desde donde quedó.
- **Gradiente:** dz/dθ usa la misma inversa con 4 pasos de refinamiento.
- **Por qué inversa y no LU:** en GPU, `lu_solve` (sustituciones triangulares) costó 17.8 ms a
  tamaño completo, contra ~1 ms de un producto matriz-vector.

**Primera medición en GPU (`tiempos/gpu/`, versión con LU):**

- F(z): 1.6 ms;
- jacobiano: 0.62 s;
- LU: 0.46 s;
- dz/dθ: 0.42 s;
- scores: 0.01 s.

La cuerda con pasos ≥ 0.01 se quedaba sin intentos (15) y caía a Newton completo desde el
inicio (~12 s). Eso daba 44-118 h de MC. De ahí los cambios de arriba.

## 1. Archivos (`claude/niu/modelo_tesis/`)

| archivo | qué tiene |
|---|---|
| `params.py` | `Config` (estático: marcas, tipos, grid de w, regímenes, tolerancias), `theta(cfg, **over)` y `regime_theta`. Las tablas de Gillingham se leen de `niu/gillingham/params.py` (una sola fuente) |
| `utils.py` | layout X/H con w, log-sumas, θ por tipo, `jacobian` por bloques |
| `bellman.py` | u(j, d, w), s(j, d, w), transición de w por intervalos (Tauchen), etapa de reparar, valores por elección, operador Γ |
| `probabilidades.py` | CCPs, núcleos físicos por edad, q estacionaria por recursión en la edad, `advance` (un periodo completo) |
| `equilibrio.py` | sistema conjunto F(z) por régimen, Newton denso, `solve_regimes`, estadísticas de mercado y por (j, a) |
| `gen_dataset.py` | panel de hogares por régimen: decisiones, r, accidentes, chatarreo endógeno, w' |
| `teoria.py` | corre todo lo anterior, compara con Gillingham y guarda CSV |
| `ll_estim.py` | núcleo de la estimación: parámetros libres (27), celdas y verosimilitud de los 3 diseños, dz/dθ con la LU reutilizada + refinamiento, scores y BHHH |
| `tiempos.py` | mide en GPU cada pieza de una evaluación de la verosimilitud y extrapola las horas del MC |
| `montecarlo.py` | MC: por réplica, panel → diseños 1, 2 y 3 (GPU) y Gillingham sobre el mismo panel (CPU, en paralelo); CSV por réplica; `--summarize` |
| `../gillingham/estimar_panel.py` | estima Gillingham (parcial y completa) sobre un panel externo; lo lanza `montecarlo.py` |
| `tests.py` | pruebas (sec. 3) |

**Convenciones:** como `niu/gillingham`.

- JAX con x64; `cfg` estático en jit; θ dinámico (un θ nuevo no recompila).
- vmap sobre los tipos de hogar.
- Sin matrices n×n en el modelo: la física va por núcleos por edad, de tamaño (A−2, J, W, W).
- La única matriz densa es el jacobiano del equilibrio, que se arma por bloques de
  `jac_chunk` columnas (`jax.linearize` + `vmap` + `lax.map`). Así la memoria en GPU queda
  acotada aunque haya ~10,000 incógnitas.

## 2. Cómo se corre

```bash
cd claude/niu/modelo_tesis
python -u tests.py                          # pruebas (segundos, CPU)
python -u teoria.py --prueba -v             # todo el flujo en chico (CPU)
python -u teoria.py -v                      # tamaño completo: a_max 25, h = 0.075 (W = 71) -> GPU
python -u teoria.py --u_w 0 --tag sin_uw -v # con u_w = 0 (o editar params.py)
```

Salidas en `claude/niu/output/modelo_tesis/teoria/<tag>/` (lista en el docstring de
`teoria.py`):

- `resumen.csv`: agregados por régimen y Gillingham;
- `por_edad.csv`: precios medios, q, keep, reparar y w medio por (j, a), con las columnas
  de Gillingham al lado;
- `panel_resumen.csv`: lo observado en el panel simulado.

**Tamaño completo por régimen:**

- W = 71 puntos de w y n = 3,411 estados por tipo;
- 10,230 incógnitas;
- jacobiano de 840 MB.

En CPU, con el grid grueso (3,174 incógnitas), cada paso de Newton tarda ~1 s. **El tiempo
por paso en GPU a tamaño completo es el dato que hace falta para planear la estimación**:
`teoria.py -v` lo imprime.

## 3. Verificado (2026-10-08, CPU)

`tests.py` (a_max = 8):

- la transición de w suma 1 y su media es exacta en el interior (= w + δ − acc_age − κr);
- el equilibrio converge;
- las CCPs y buy suman 1;
- q = qM con error de 1e-16 (con `advance`, sin armar M);
- flujo estacionario por marca;
- Pr(reparar) = 0 en la edad A−1;
- el panel simulado reproduce keep y Pr(reparar) del modelo (±0.002).
- **gradiente implícito de la LL contra diferencias finitas**, diseños 1, 2 y 3: error
  relativo ≤ 2e-6, usando la LU de un θ cercano.
  - Con 2 pasos de refinamiento, dz/dθ queda a 1e-5; con 5, a 6e-11. Default: 4.
  - Dos arreglos que salieron de esta prueba:
    - los bordes extremos de la transición de w son ±1e3 y no ±∞ (con ∞, la derivada
      respecto a σ_η daba 0·∞ = NaN);
    - el factor de compra es `where(trade, buy, 1)` y no `buy ** trade` (0⁰ tiene derivada
      NaN).
- `tiempos.py --prueba` (a_max = 8): la cuerda converge sin refactorizar para pasos de hasta
  0.03 en x (4-12 pasos).

**Solver:** desde cero, el régimen central converge en 14 pasos de Newton (tope de 150
mil DKK por paso en precios). Los regímenes ±0.3, arrancando del central, convergen en 5.

## 4. Resultados preliminares

Grid grueso (h = 0.25, W = 22) y a_max = 25, CPU. Es solo una vista: el grid aprobado es
h = 0.075. Carpeta `output/modelo_tesis/teoria/preliminar_h025/`.

**Agregados:**

| | ζ = −0.3 | ζ = 0 | ζ = +0.3 | Gillingham |
|---|---|---|---|---|
| sin coche | 0.322 | 0.335 | 0.349 | 0.336 |
| compran nuevo / usado | .037 / .196 | .038 / .198 | .039 / .201 | .038 / .191 |
| reparan (por coche en mano) | 0.40 | 0.31 | 0.22 | — |
| chatarreo endógeno / accidentes | .030 / .021 | .027 / .026 | .024 / .032 | .026 / .031 |
| edad media del parque | 10.2 | 9.9 | 9.5 | 9.6 |
| masa en los bordes de w | 0.5% | 0.2% | 0.2% | — |

**Por edad, light brown, régimen central contra Gillingham** (promedio sobre w):

| edad | 1 | 2 | 5 | 10 | 15 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| P medio (stock) | 140.2 | 128.1 | 95.3 | 57.9 | 25.6 | 24.9 | 20.8 |
| P Gillingham | 139.1 | 125.1 | 88.6 | 49.8 | 19.5 | 22.5 | 21.8 |
| q / q Gillingham | .032/.030 | .032/.030 | .031/.029 | .029/.027 | .024/.022 | .012/.008 | .004/.002 |
| Pr(reparar) | 0.69 | 0.66 | 0.54 | 0.32 | 0.15 | 0.08 | 0 |
| w medio | 0.08 | −0.02 | −0.27 | −0.47 | −0.54 | −0.89 | −1.28 |

**Lectura:**

1. **La calibración objetivo casi se cumple sin ajustar nada:**
   - hogares sin coche y compras iguales a Gillingham;
   - distribución por edad parecida;
   - precios medios que bajan igual, aunque light brown queda 6-8 mil DKK arriba en edades
     medias (sus coches están mejor que el promedio de Gillingham, porque w medio < 0).
2. **R mueve mucho la reparación:** 0.40 → 0.31 → 0.22 entre regímenes. Eso es lo que
   identifica en Hu & Xin.
3. **La masa en los bordes de w queda debajo del 0.5%.** Pasa el diagnóstico.
4. **Problema: la reparación de light brown baja con la edad desde 0.69.** No cumple el
   criterio de poca reparación al inicio (`modelo_fin.md`, sec. 4.2).
   - Con κ constante, reparar le da al coche nuevo el mismo beneficio por año que al
     viejo (u_w κ por año más supervivencia), y el nuevo lo disfruta más años.
   - Heavy brown repara poco en todas las edades (0.42 en la edad 1 → 0.04 en la 10) porque
     R es 2.5 veces mayor.
   - Al final de la vida sí repara poco.

## 5. Reparar poco en coches jóvenes  [decidido: con R(LB, a)]

**Decisión del autor:** la transición no se toca; el patrón viene de los precios de
reparación. La opción "ρ" (fracción del desgaste acumulado) se probó el 2026-10-08 y se
descartó.

Se usa R(LB, a) = 4.0 + 0.10 (a − 1) + 6.0 e^{−(a−1)/4}: caro de joven y casi plano después.

**Vista preliminar** (grid grueso, `teoria/preliminar_RLB_h025/`), light brown:

| edad | 1 | 3 | 5 | 7 | 10 | 15 | 20 | 23 |
|---|---|---|---|---|---|---|---|---|
| Pr(reparar), ζ = 0 | .26 | .37 | .42 | .43 | .38 | .26 | .17 | .12 |
| Pr(reparar), ζ = ∓0.3 | .44/.10 | .53/.19 | .56/.26 | .55/.27 | .50/.24 | .35/.15 | .25/.10 | .19/.07 |
| P medio | 139.2 | 114.2 | 93.6 | 75.3 | 58.5 | 27.9 | 26.5 | 16.0 |
| P Gillingham | 139.1 | 111.8 | 88.6 | 68.3 | 49.8 | 19.5 | 22.5 | 15.7 |

**Lectura:**

- La reparación sube y luego baja con la edad.
- La distribución por edad es igual a la de Gillingham.
- Los precios de light brown caen menos que en Gillingham: 9 mil DKK arriba a la edad 10.

Opciones que se consideraron antes (descartadas):

1. **El efecto de reparar crece con el desgaste acumulado** (recomendada):

       ℓ' = ℓ + δ − r · ρ · (ℓ − ℓ_nuevo) + η

   Reparar recupera una fracción ρ de lo que el coche se ha desgastado desde nuevo.
   - Un coche joven casi no tiene qué recuperar.
   - Un coche gastado recupera mucho, así que se repara más cuando w es alto. Eso también
     da variación de p en w.
   - Hu & Xin permiten que el efecto dependa del estado: m(1, x) − m(0, x) puede variar
     con x.
   - Se mantiene la persistencia si ρ < 1: un coche reparado nunca vuelve a nuevo.
2. **κ crece con la edad:** κ_a = κ · min(a / a*, 1). Es más simple, pero no depende del
   estado del coche.
3. **R alto en edades jóvenes** (por ejemplo, el servicio de agencia en garantía). Es ad
   hoc.

## 6. Para la estimación (siguiente paso)

**Meta: < 10 horas** con 2 GPUs:

- 50 réplicas × 4 estimaciones (3 diseños + Gillingham) = 200 estimaciones;
- ~6 min de GPU por estimación con 3 regímenes.

**El costo lo domina armar el jacobiano del equilibrio** (10,230 columnas por régimen).
Plan:

- **Arranque en caliente y Newton "de cuerda":** reutilizar la factorización LU de la
  evaluación anterior (los θ de L-BFGS cambian poco) y solo rehacerla cuando la
  convergencia se frene.
- **Gradiente implícito con esa misma LU:** dz/dθ = −F_z⁻¹F_θ, con 30 columnas y unos
  pasos de refinamiento con jvp exactos.
- **El estimador de Gillingham es barato:** 150 incógnitas, segundos.

Decidir n_w y el plan con el tiempo por paso de Newton medido en GPU (`teoria.py -v`).

## 7. Monte Carlo (`montecarlo.py`)

**Por réplica:**

1. Simula el panel: 10,000 hogares por régimen, 4 años, 3 regímenes.
2. Lanza Gillingham en CPU sobre el mismo panel, sin w ni r (estados (j, a)), con las
   verosimilitudes parcial y completa.
3. Mientras tanto, la GPU estima los diseños 1, 2 y 3. Arrancan desde la verdad, con los
   equilibrios verdaderos y sus inversas.
   - Cada evaluación: cuerda desde el predictor, dz/dθ y scores.
   - En la réplica `--rep_perturbados` (0), además 2 arranques perturbados.
4. Escribe los CSV e imprime una línea por diseño y la línea `[avance]`.

**Cómo correrlo:**

```bash
cd claude/niu/modelo_tesis
python -u montecarlo.py --smoke                            # juguete, CPU (~2 min)
python -u montecarlo.py --reps 0:1 -v | tee mc_rep0.log    # una réplica completa (GPU), con log de L-BFGS
CUDA_VISIBLE_DEVICES=0 nohup python -u montecarlo.py --reps 0:25  > mc_0-24.log  2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u montecarlo.py --reps 25:50 > mc_25-49.log 2>&1 &
python -u montecarlo.py --summarize                        # resumen_mc.csv
```

- **El MC grande y la réplica 0 se pisan.** Corre `--reps 0:25` solo si la réplica 0 de
  prueba se borra o va en otro `--tag`. Si no, la réplica 0 queda dos veces en los CSV.
- **Etiquetas de estimador:** `diseno_1`, `diseno_2`, `diseno_3`, `gill_parcial` y
  `gill_completa`. Los nombres de parámetro coinciden (`mu_t0`, `u0_t1_j0`, `acc_age_j0`,
  ...): la tabla resumen compara el sesgo de Gillingham contra los diseños parámetro por
  parámetro.

**Verificado:** `--smoke` corre completo en CPU (2 min). Escribe los 4 CSV y la tabla
resumen.
