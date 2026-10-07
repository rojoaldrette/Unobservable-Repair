# CLAUDE.md

Proyecto de tesis (Rodrigo Aldrette, COLMEX). Responde en español.

## Qué es

Modelo de equilibrio estacionario del mercado de autos basado en Gillingham, Iskhakov,
Munk-Nielsen, Rust & Schjerning, con una decisión **no observada** de reparar,
identificada con la estrategia de Hu & Xin (2024, J. Econometrics), "Identification and
estimation of dynamic structural models with unobserved choices".

Objetivo: tomar parámetros estimados de Gillingham, añadir precios de reparación, resolver
el equilibrio (P, q) y hacer Monte Carlo para ver qué tanto se infieren las decisiones de
reparación no observadas.

## Modelo

- Horizonte infinito, un periodo = un año. Precios en miles de DKK, utilidades en utils.
- Estados: marca j (J = 3), edad a (1..a_max-1 activas, a_max = 7 terminal), s = probabilidad
  de descomponerse (grid de n_s = 100 en [s_min, s_max]). Más el estado "sin coche".
- Terminal: el coche desaparece; s no importa. Coches nuevos nacen todos en s_new (ajustado
  al grid con `s_new_index`).
- Decisiones de un dueño de coche activo: keep, repair (keep + reparar), purge (vender y
  quedarse sin coche), trade (deshacerse del coche y comprar otro nuevo o usado).
- Descompostura: con prob s el coche pasa a terminal. Si sobrevive:
  s' = m(r, j, a, s) + eta, eta ~ N(0, s_sigma^2) discretizado (Hu & Xin, Assumption 2).
- Identificación: keep/trade se observa, repair no (observabilidad parcial, Hu & Xin
  sec. 6.3, ec. 6.4). Variable excluida: precio de reparación R(j, a) por año
  (Assumption 7): mueve CCPs, no la transición de s.
- Equilibrio: precios del mercado secundario P(j, a, s), forma (J, a_max-1, n_s), y
  distribución estacionaria q.
- Errores: valor extremo tipo I (logit simple). Nested logit disponible bajando sigma_trade
  y sigma_sell (requiere 0 < sigma_sell <= sigma_trade <= sigma).

## Código (`scripts/`)

- `params.py`: `Params`, dataclass frozen. Todos los vectores son tuplas para que sea
  hashable.
- `bellman.py`: grids, primitivas, operador `T`, solver, CCPs, matrices de transición,
  distribución estacionaria.

Flujo: `solve_bellman(P, g)` → `ccps(EV, P, g)` → `transition_matrix(EV, P, g)` →
`stationary_distribution(M)`. `trade_matrices` da Ω desagregada (keep, repair, trade,
purge); `physical_matrices` da Q_0 y Q_1. `observed_kept_transition` da la mezcla que ve
el econometrista.

## Layout de estados (importante)

Vector de largo n = J*(a_max-1)*n_s + J + 1 (= 1804 con defaults):

- X (inicio de periodo): `[act(j, a, s), a=1..a_max-1 | term(j) | none]`
- H (tenencia post-comercio): `[used(j, d, s), d=1..a_max-1 | new(j) | none]`

El hueco del terminal en X lo ocupa el coche nuevo en H (truco de Gillingham), así que
Ω (X → H) y Q (H → X) son cuadradas. Usar `split_states` y `stack_states`, no índices a mano.

EV vive en X y no tiene eje r. `continuation_values` devuelve cv_used (2, J, a_max-1, n_s),
condicional en la elección r, indexado por la tenencia h.

## Convenciones

- JAX con `jax_enable_x64 = True` (las tolerancias de 1e-12 lo necesitan).
- `g` (Params) siempre es argumento estático: `@partial(jax.jit, static_argnames="g")`.
  P y EV son dinámicos.
- Lo que depende solo de `g` y debe ser entero estático (p. ej. `s_new_index`) se calcula
  con numpy, no con jnp.
- Nada de control de flujo en Python sobre arrays dentro de funciones jiteadas.
- No usar `.at[]` cuando `concatenate`/`where` basten.
- Solver: successive approximations con `jaxopt.FixedPointIteration` hasta `sa_tol`,
  luego Newton-Kantorovich con jacobiano analítico dT/dEV = beta * M (válido para GEV).
- La distribución de errores solo entra por `emax`, `choice_probs` y los sigmas.
- Para cambiar de "año" (otro vector de precios de reparación):
  `dataclasses.replace(g, repair_price=...)`.

## Decisiones y pendientes

- Transición de s implementada: **opción 2** (reparar solo mueve la media de s'; la prob de
  descomponerse hoy es el s actual). La alternativa, "opción 1 reparar", está en
  `opcion_1_reparar.md`: reparar baja la prob de descomponerse ya este periodo (implica
  selección por supervivencia en los pesos de la mezcla). Pendiente decidir según cuándo se
  mide s en los datos.
- Chatarreo endógeno: verificar en el código si ya se quitó (solo vender al deshacerse de
  un coche activo: `disposal = mu * P - tc_sell`).
- Pendiente: ¿un dueño de terminal puede comprar ese mismo periodo o pasa un periodo sin coche?
- Pendiente (lo escribe el autor): demanda y oferta agregadas y solver de equilibrio
  ED(P) = 0. Fórmulas sugeridas al final de `bellman.py`.
- Parámetros de Gillingham usados: hogar "Low WD, Couple, Poor"; marcas light brown,
  light green, heavy brown. Interpretación de la Tabla 10 (costos de transacción) por
  verificar. beta = 0.95, u2 = 0, chatarra, reparación y dinámica de s son calibraciones
  propias.
- Cuidado: en las fronteras del grid de s, E[eta | s] != 0 por el recorte (viola el
  supuesto de Hu & Xin).

## Verificación

`python scripts/bellman.py` corre pruebas: filas de Ω y Q suman 1, M rápida == M densa,
q válida, CCPs de reparación por edad. El código original solo se probó con un shim de
numpy; confirmar con JAX real.


## Reglas

Solo puedes crear archivos y modificar archivos en el directorio `claude/`,`.claude/` y `output/claude/`, fuera de esas carpetas no puedes borrar, modificar o crear archivos. No puedes tampoco leer ningún archivo fuera de este repositorio, tu existencia se limita dentro de la carpeta tesis_final_1. Te dejé los papers en docs para que no tengas que buscar nada.

En la carpeta `claude/` tendrás tu propio `scripts/` y `docs/`. Tienes derechos de read a todo el repositorio.

En `scripts/` harás tus propios archivos que también iré modificando yo. En `docs/` tendrás tu propia documentación y comentarios indepth de tus scripts.

En `output/claude/` tendrás la carpeta `graphs/` y `montecarlo/` (Montecarlo datasets) para tu propio uso.
