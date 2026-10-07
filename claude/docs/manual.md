# Manual de uso

Cómo correr todo el código de `claude/scripts/`: equilibrios, Monte Carlo, estimaciones,
gráficas y la supercomputadora.  Para cómo está hecho por dentro, ver `codigo.md`; para
el detalle de los estimadores y las columnas de cada CSV, `reporte_estimacion.md`; para
los valores de la calibración, `calibracion.md`.

Contenido:
0. Antes de empezar: rutas y archivos con el mismo nombre
1. Instalación
2. modelo_fin con distintas calibraciones
3. Gillingham con distintas calibraciones
4. Monte Carlo y estimaciones
5. Resultados y gráficas
6. Supercomputadora (2 × Quadro GV100), paso a paso
7. Problemas comunes
8. Dónde está cada documento

---

## 0. Antes de empezar: rutas y archivos con el mismo nombre

**Todas las rutas de este manual son relativas a la raíz del repo** (`tesis_final_1/`),
salvo que diga otra cosa.  Hay varios archivos con el mismo nombre en carpetas distintas;
esta es la lista de **los que se corren** y qué hace cada uno:

| archivo (ruta completa desde la raíz) | qué hace | ¿actual? |
|---|---|---|
| `claude/scripts/modelo_fin/estimar.py` | **modelo de la tesis**: equilibrios verdaderos, panel simulado, estimación D0 (`oraculo`, r observada) y D1 (`hx`, r no observada) | **sí, el principal** |
| `claude/scripts/gillingham/estimar.py` | **réplica de Gillingham**: estimación sobre sus propios datos, o sobre el panel de modelo_fin (`--panel`, el "cruce") | **sí** |
| `claude/scripts/analisis/main.py` | gráficas y tablas a partir de las carpetas que escriben los dos `estimar.py` | **sí** |
| `claude/scripts/comparacion/compare.py` | resuelve los equilibrios G25 (Gillingham) y M25 (modelo_fin, 1 tipo) y los compara en un CSV, sin estimar | sí (rápido, sin estimación) |
| `claude/scripts/modelo_fin/tests.py` | pruebas del modelo de la tesis | sí |
| `claude/scripts/gillingham/tests.py` | pruebas de la réplica | sí |
| `claude/scripts/correr_gpu.sh` | **todo en la supercomputadora** por fases (`prueba`, `fase1`, ..., `todo`), repartido en las 2 GPUs | **sí** (sección 6.0) |
| `claude/scripts/slurm_estimar.sh` | plantilla para un clúster **con SLURM** (`sbatch`) | solo si hay SLURM (ver 6.9) |
| `claude/scripts/modelo_fin/main.py` | MC **viejo** (v1.0): solo la primera etapa de Hu & Xin (transición de s), no la estimación estructural | histórico |
| `claude/scripts/modelo_fin/slurm_mc.sh` | SLURM para el MC viejo de arriba | histórico |
| `claude/scripts/gillingham/main.py` | MC **viejo** de la réplica (v1.3-v1.4), escribe en `claude/output/montecarlo/gillingham/` | histórico (lo reemplaza `gillingham/estimar.py`) |
| `scripts/main/main.py`, `scripts/main/*.py` | **código del autor** en la raíz del repo, no es parte de `claude/` | no confundir |

Los demás `.py` (`params.py`, `bellman.py`, `theta.py`, ...) son módulos: no se corren,
se importan.  `modelo_fin/` y `gillingham/` tienen módulos con el mismo nombre
(`params.py`, `bellman.py`, `equilibrium.py`, ...) pero son **distintos**; por eso:

> **Regla de oro:** cada script se corre **desde su propia carpeta**
> (`cd claude/scripts/modelo_fin` antes de `python estimar.py`).  Los imports son
> locales: si corres `python claude/scripts/modelo_fin/estimar.py` desde otra carpeta,
> Python puede no encontrar `params` o, peor, importar el `params.py` de otra carpeta.

Dónde escribe cada cosa (todas las rutas desde la raíz):

| quién | escribe en |
|---|---|
| `modelo_fin/estimar.py` | `claude/output/estimaciones/modelo_fin/<tag>/` |
| `gillingham/estimar.py` | `claude/output/estimaciones/gillingham/<tag>/` |
| `analisis/main.py` | `claude/output/analisis/<nombre>/graficas/` y `.../tablas/` |
| `comparacion/compare.py` | `claude/scripts/comparacion/output/` (no se sube a git) |
| `modelo_fin/main.py` (viejo) | `claude/scripts/modelo_fin/output/` (no se sube a git) |
| `gillingham/main.py` (viejo) | `claude/output/montecarlo/gillingham/` |

---

## 1. Instalación

Python 3.11 o más reciente.

```bash
# Laptop (CPU): solo para pruebas chicas
pip install jax jaxopt scipy pandas matplotlib

# Supercomputadora (GPU NVIDIA, CUDA 12): ver sección 6.2 (entorno virtual y verificación)
pip install -U "jax[cuda12]" jaxopt scipy pandas matplotlib
```

Variables de entorno que conviene tener **siempre**:

```bash
export PYTHONIOENCODING=utf-8      # hay acentos en los mensajes
```

y correr con `python -u` cuando la salida va a un archivo de log (si no, el log queda
vacío hasta que termina el programa).

Prueba rápida de que todo funciona (minutos, tamaño de juguete):

```bash
cd claude/scripts/modelo_fin
python -u tests.py --n_s 12 --estructural
```

`tests.py` sin opciones prueba la mecánica con `Params()` (el modelo viejo: a_max = 7, s en
niveles); `--estructural` prueba la calibración "tesis" (log-odds, chatarreo) a tamaño de
juguete; `--ed` también resuelve un equilibrio con la API vieja.

---

## 2. modelo_fin con distintas calibraciones

### 2.1 Qué es "una calibración" en este código

Los números que definen un equilibrio de modelo_fin salen de **cuatro lugares**.  Conviene
tenerlos claros porque se cambian de forma distinta:

| capa | qué fija | dónde se cambia |
|---|---|---|
| **1. calibración con nombre** | todo lo común a los hogares: transición de s (log-odds o niveles), grid, R(j, a), sigma_repair, chatarreo, Ts, p_new, beta, ... | `--calib` (`tesis`, `tesis_v0`, `gill_s`, `defaults`) o una calibración nueva en `claude/scripts/modelo_fin/calibracion.py` (2.4) |
| **2. tamaño** | a_max (edad terminal) y n_s (puntos del grid de s) | `--a_max`, `--n_s` |
| **3. tipos de hogar** | **mu, u0, u1, tc_buy, tc_buy_nocar** de cada tipo y la fracción de cada tipo | `--types`, `--f`; los valores están en `PAPER_TYPES` (`claude/scripts/modelo_fin/params.py`) |
| **4. años de R** | cuántos "años" (regímenes) y cuánto varía R entre años: R_t = R · exp(zeta_t), zeta_t en [−spread, spread] | `--T`, `--spread` |

> **Ojo con la capa 3.** En el código actual (`theta.py`, `equilibrium.py`, `estimar.py`)
> mu, u0, u1, tc_buy y tc_buy_nocar **se toman de `PAPER_TYPES`, no de `Params`**.  Hacer
> `dataclasses.replace(g, mu=0.12)` **no cambia nada** en `estimar.py` ni en
> `equilibrium.solve`; solo afecta a la API vieja (`ED.solve_equilibrium(g)`, un tipo).
> Para cambiarlos, ver 2.5.

Las calibraciones con nombre (`claude/scripts/modelo_fin/calibracion.py`):

| `--calib` | s | a_max pensado | chatarreo | R(j, a=1) | para qué |
|---|---|---|---|---|---|
| **`tesis`** (default) | **log-odds**: ℓ = logit(s), ℓ' = ℓ + acc_age_j − 0.5 r + η | 25 | sí | (4, 6.5, 10) mil DKK | **la de la tesis**; mismos parámetros que Gillingham (`calibracion.md`) |
| `tesis_v0` | niveles, calibrada a la Tabla 4 en edades bajas | 7 | no | (4, 6.5, 10) | default de v1.7 |
| `gill_s` | niveles, como `tesis_v0` | 7 | no | (15, 12.5, 25) | M7_gill de `compare.py --viejos` |
| `defaults` | niveles, `Params()` | 7 | no | (15, 12.5, 25) | el modelo original |

**¿Se usan los log-odds?** Sí, en `tesis` (y por lo tanto en todos los corredores, porque
es el default).  Las otras tres siguen en niveles.  Para comprobarlo en una corrida: el
`config.json` de la carpeta de salida dice `"s_space": "logodds"`, y `precios.csv` trae la
columna `ell` (= logit(s)) además de `s`.  Detalle en `codigo.md`, sección 11.

### 2.2 Resolver solo el equilibrio (sin estimar), desde Python

Para ver precios, distribución y CCPs de una calibración sin correr ninguna estimación.
Abrir Python **en `claude/scripts/modelo_fin/`**:

```bash
cd claude/scripts/modelo_fin
python
```

```python
import numpy as np
from params import Types
from calibracion import calibracion
from theta import economy
from equilibrium import solve, split_z, equilibrium_objects
from utils import make_s_grid, split_states

# 1. Elegir calibración y tamaño (n_s = 40 para la laptop; 100 en la GPU)
g = calibracion("tesis", a_max=25, n_s=40)

# 2. Elegir tipos de hogar (default: pareja pobre + soltero pobre, mitad y mitad)
types = Types()                                   # o Types(("low_couple_poor",), (1.0,))
eco = economy(g, types)

# 3. Resolver el equilibrio: z = (EV de cada tipo, P)
z, convergio, iteraciones = solve(eco, method="krylov", verbose=True)
EVs, P = split_z(z, eco)                          # P: (J, a_max-1, n_s), miles de DKK

# 4. Objetos de equilibrio por tipo (numpy): P, EV, q, keep, purge, trade, buy, repair, scrap
objs = equilibrium_objects(z, eco)
q_act, q_term, q_none = split_states(objs[0]["q"], g)        # vector de estados -> (J, a_max-1, n_s), (J,), ()
rep_act, _, _ = split_states(objs[0]["repair"], g)          # Pr(reparar | coche usado h)
print("sin coche, tipo 0:", q_none)
print("Pr(reparar), marca 0, edad 1, por s:", rep_act[0, 0])
s = np.asarray(make_s_grid(g))                    # prob. de descomponerse en cada punto del grid
```

Otro "año" de precios de reparación (misma economía, otro R):

```python
R = np.asarray(g.repair_price) * 1.3              # forma (J, a_max-1)
z2, ok, it = solve(eco.with_repair_price(R), z0=z) # z0 = arranque en caliente
```

Para la comparación rápida de un solo tipo contra Gillingham (G25 contra M25 y M25 sin
chatarreo, solo equilibrios, CSV):

```bash
cd claude/scripts/comparacion
python -u compare.py --n_s 40            # 100 en la GPU
# salida: claude/scripts/comparacion/output/comparacion.csv y comparacion_resumen.csv
```

### 2.3 Cambiar la calibración desde la línea de comandos

`modelo_fin/estimar.py` (sección 4) acepta las capas 1-4 directamente:

```bash
cd claude/scripts/modelo_fin
python -u estimar.py --calib tesis    --a_max 25 --n_s 100 --types low_couple_poor,low_single_poor --T 13 --spread 0.3 ...
python -u estimar.py --calib tesis    --a_max 25 --n_s 100 --types low_couple_poor --f 1.0 ...         # un solo tipo
python -u estimar.py --calib tesis    --types low_couple_poor,low_single_poor --f 0.7,0.3 ...          # 70/30
python -u estimar.py --calib tesis_v0 --a_max 7  --n_s 100 ...                                        # la de v1.7
```

Tipos disponibles (`PAPER_TYPES`): `low_couple_poor`, `low_couple_rich`,
`low_single_poor`, `low_single_rich`.  Recordatorio: para separar tc_buy de tc_sell
sin precios hacen falta **al menos dos tipos con mu distinta** (pareja pobre 0.113,
soltero pobre 0.094); con un solo tipo esa dirección no está identificada.

### 2.4 Una calibración nueva con nombre (para usarla con `--calib`)

Cualquier cosa de la capa 1 que no esté en las opciones de línea de comandos (s_repair,
s_sigma, sigma_repair, R, chatarreo, grid de s, ...) se cambia agregando una calibración
a `claude/scripts/modelo_fin/calibracion.py`.  Ejemplo: "tesis" con reparación más
efectiva y R más caro.

1. En `calibracion.py`, agregar el nombre a `NOMBRES`:

   ```python
   NOMBRES = ("tesis", "tesis_rep_alta", "tesis_v0", "gill_s", "defaults")
   ```

2. En la función `calibracion`, justo después del bloque `if nombre == "tesis": ...`:

   ```python
   if nombre == "tesis_rep_alta":
       return dataclasses.replace(
           calibracion("tesis", a_max, n_s),
           s_repair=0.8,                                              # reparar baja las odds 55%
           repair_price=repair_prices(a_max, (6.0, 9.0, 14.0), 0.06)) # R más caro
   ```

3. Usarla: `python -u estimar.py --calib tesis_rep_alta ...`.  El nombre entra en el tag
   de la carpeta de salida, así que no se mezcla con las corridas de `tesis`.

Campos más usados de `Params` (`claude/scripts/modelo_fin/params.py`; con `tesis`, las
unidades de s_* son de ℓ = logit(s)):

| campo | qué es | en `tesis` |
|---|---|---|
| `s_repair` | cuánto baja ℓ' al reparar | 0.5 |
| `s_sigma` | sd del ruido η de la transición | 0.15 |
| `s_const` | deriva anual de ℓ por marca | acc_age_j (Tabla 4) |
| `s_new` | ℓ de un coche nuevo por marca | acc_int_j (Tabla 4) |
| `s_min`, `s_max`, `n_s` | grid de ℓ | −9, 1, 100 |
| `repair_price` | R(j, a), tupla J × (a_max − 1) | `repair_prices(a_max, (4, 6.5, 10), 0.06)` |
| `sigma_repair` | escala del shock de la decisión de reparar | 0.3 |
| `scrap`, `sigma_sell` | chatarreo endógeno y su escala | True, 0.3454 |
| `u_s` | desutilidad de un coche frágil (por unidad de s) | 0 |
| `tc_sell`, `tc_sell_inspect` | costo de vender (normal / año de inspección) | 0.9106, 2.1929 |
| `p_new`, `p_scrap`, `beta` | precios de nuevos y chatarra, descuento | Tabla 3, 2, 0.95 |

`repair_price` **tiene que tener a_max − 1 columnas**: por eso se arma con
`repair_prices(a_max, ...)` dentro de la función y no como tupla fija.

Recordatorio del criterio de calibración (`calibracion.md`): solo los **parámetros**
deben coincidir con Gillingham; no se calibra s_repair, s_sigma, sigma_repair ni R para
acercar los resultados a G25.

### 2.5 Cambiar mu, u0, u1, tc_buy o tc_buy_nocar (los parámetros por tipo)

Dos opciones:

- **Permanente (sirve para `estimar.py`):** editar o agregar un tipo en `PAPER_TYPES`
  (`claude/scripts/modelo_fin/params.py`) y usarlo con `--types`.  Si se agrega un tipo
  nuevo, ponerlo también en `claude/scripts/gillingham/params.py` si se va a correr el
  cruce (los dos `PAPER_TYPES` son copias y tienen que coincidir).
- **Solo en Python:** modificar el diccionario θ antes de armar la economía.

  ```python
  import jax.numpy as jnp
  from theta import economy, theta_types
  th = theta_types(g, types)                  # dict; los campos por tipo tienen eje 0 = tipo
  th["mu"] = jnp.array([0.12, 0.09])
  eco = economy(g, types, th)
  ```

---

## 3. Gillingham con distintas calibraciones

La réplica (`claude/scripts/gillingham/`) **no tiene `--calib`**.  Su calibración es:

| capa | dónde | se cambia con |
|---|---|---|
| parámetros comunes: Ts, sigma_sell, accidentes (Tabla 4), p_new, p_scrap, beta | `GParams` en `claude/scripts/gillingham/params.py` | editar los defaults de `GParams` (o Python, abajo) |
| tamaño | `GParams.a_max` (25 = el paper) | `--a_max` |
| parámetros por tipo (mu, u0, u1, tc_buy, tc_buy_nocar) | `PAPER_TYPES` en el mismo archivo | `--types`, `--f`, o editar `PAPER_TYPES` |

Los parámetros son los mismos que usa modelo_fin con `tesis`; la diferencia es que aquí
no hay s ni reparación: la prob. de accidente es directamente
logit(acc_int_j + acc_age_j a).

**Resolver solo el equilibrio, desde Python** (en `claude/scripts/gillingham/`):

```python
import dataclasses
from params import GParams, GTypes
from theta import economy
from equilibrium import solve, split_z

g = dataclasses.replace(GParams(), a_max=25)              # otra calibración: replace(..., tc_sell=1.2)
types = GTypes(("low_couple_poor", "low_single_poor"), (0.5, 0.5))
eco = economy(g, types)
z, ok, it = solve(eco, verbose=True)
EVs, P = split_z(z, eco)                                  # P: (J, a_max-1)
```

Mismo cuidado que en modelo_fin: mu, u0, u1, tc_buy y tc_buy_nocar salen de
`PAPER_TYPES`, no de `GParams`.

**Desde la línea de comandos** (`gillingham/estimar.py`, sección 4.4): `--a_max`,
`--types`, `--f`.  Cualquier otro cambio (p. ej. tc_sell) hay que hacerlo en los defaults
de `GParams`, porque `estimar.py` arma `g = replace(GParams(), a_max=...)`.  Si se cambia,
anotarlo: el `config.json` de la corrida guarda los parámetros usados.

**Comparación de equilibrios G25 vs M25** (un tipo, sin estimar): `compare.py`, ver 2.2.

---

## 4. Monte Carlo y estimaciones

### 4.1 Qué se estima

Un Monte Carlo es: fijar la "verdad" (una calibración), resolver sus equilibrios,
simular paneles con semillas distintas (una por réplica) y estimar en cada panel.  Hay
tres ejercicios:

| ejercicio | script | datos | estimadores (columna `estimador`) |
|---|---|---|---|
| **A. modelo_fin** | `modelo_fin/estimar.py` | panel de modelo_fin (con reparación) | `oraculo` (D0: r observada), `hx` (D1: r no observada, Hu & Xin) |
| **B. Gillingham propio** | `gillingham/estimar.py` | panel de Gillingham (sin reparación) | `gill_parcial` (la verosimilitud del paper: sin precios ni accidentes), `gill_completa` (oráculo) |
| **C. Cruce (el sesgo)** | `gillingham/estimar.py --panel` | panel de modelo_fin, sin s ni r | `gill_parcial`, `gill_completa` (en el análisis aparecen como `...|modelo_fin`) |

La pregunta de la tesis: D1 (`hx`) recupera la verdad casi como D0 (`oraculo`), mientras
que Gillingham sobre datos con reparación (C) está sesgado.

Cada réplica de A: simula T = 13 años × N = 20,000 hogares × K = 2 años por hogar,
estima D0 y D1 con L-BFGS + BHHH desde la verdad y `--n_starts` arranques perturbados, y
se queda con el de mayor verosimilitud (guarda todos).

### 4.2 Las reglas que hay que respetar

1. **`--reps a:b` corre las réplicas a, a+1, ..., b−1** (la semilla es el número de
   réplica).  `--reps 0:1` es solo la réplica 0.
2. **Siempre pasar `--tag`** con un nombre fijo.  Todos los bloques de un mismo Monte
   Carlo deben ir al mismo tag **y con exactamente las mismas opciones de diseño**
   (`--calib --a_max --n_s --T --spread --N --K --types --f --fix`).  El análisis junta
   todos los archivos `parametros_reps*.csv` de la carpeta.
3. **Primero la réplica 0, sola.**  Resuelve los equilibrios verdaderos y los guarda en
   `<tag>/verdad.npz`; los demás bloques lo reutilizan (si el diseño coincide).  Si se
   lanzan varios bloques antes de que exista `verdad.npz`, cada uno lo resuelve por su
   cuenta (tiempo perdido y escrituras simultáneas).
4. **En los bloques siguientes, `--rep_reporte 0`.**  `precios.csv`, `distribucion.csv`
   y `ccps.csv` se escriben para una sola réplica (la primera del bloque si no se dice
   otra cosa); sin `--rep_reporte 0`, cada bloque sobrescribe los de la réplica 0.
5. **`--guardar_panel`** guarda `panel_rep<k>.csv.gz`.  Lo necesita el cruce (C).  Para
   el cruce de una sola réplica basta en la réplica 0; para el MC del cruce, en todas.
   Los paneles no se suben a git (`claude/.gitignore`).
6. Si un bloque se cae, lo hecho queda: los CSV se escriben después de cada réplica.
   Para continuar, relanzar desde la réplica que falta con el mismo tag (p. ej. si
   `parametros_reps10-19.csv` llega hasta la 14, correr `--reps 15:20`).

### 4.3 Ejercicio A: modelo_fin paso a paso

Desde `claude/scripts/modelo_fin/`.

```bash
cd claude/scripts/modelo_fin
export PYTHONIOENCODING=utf-8

# 0. Prueba de humo (tamaño de juguete, minutos; sale en .../modelo_fin/smoke/)
python -u estimar.py --smoke

# 1. Réplica 0: verdad + D0 + D1 + panel
DISENO="--calib tesis --a_max 25 --n_s 100 --T 13 --spread 0.3 --N 20000 --K 2 --types low_couple_poor,low_single_poor"
TAG_MF=tesis_A25_S100_T13_N20000_K2
python -u estimar.py --reps 0:1 $DISENO --tag $TAG_MF --guardar_panel -v

# 2. Más réplicas, por bloques (en paralelo en la GPU, ver sección 6)
python -u estimar.py --reps 1:50   $DISENO --tag $TAG_MF --rep_reporte 0 --guardar_panel
python -u estimar.py --reps 50:100 $DISENO --tag $TAG_MF --rep_reporte 0 --guardar_panel
```

Otras opciones útiles:

| opción | default | para qué |
|---|---|---|
| `--infos` | `oraculo,hx` | solo uno de los dos estimadores (`--infos hx`) |
| `--fix` | — | dejar parámetros fijos en la verdad: `--fix u_s,s_persist` (nombres de `theta.FIELDS`: mu, u0, u1, u_s, tc_buy, tc_buy_nocar, tc_sell, tc_sell_inspect, sigma_sell, sigma_repair, s_const, s_age, s_persist, s_repair, s_sigma) |
| `--n_starts` | 2 | arranques perturbados además de la verdad (más = más lento, más seguro contra máximos locales) |
| `--perturb` | 0.1 | tamaño relativo de la perturbación de los arranques |
| `--method` | `krylov` | `dense` solo para tamaños chicos (arma matrices n × n) |
| `--lbfgs_iter`, `--bhhh_iter` | 500, 50 | iteraciones de cada optimizador |
| `-v` | no | imprime el avance |

Qué sale en `claude/output/estimaciones/modelo_fin/<tag>/`:

| archivo | contenido |
|---|---|
| `config.json` | opciones, todos los Params y los tipos (para saber después qué se corrió) |
| `verdad.npz` | equilibrios verdaderos de los T años (se reutiliza) |
| `parametros_reps<a>-<b>.csv` | por réplica, estimador, arranque y parámetro: verdad, estimado, se, z, p, IC 95%; `mejor = True` marca el arranque elegido |
| `resumen_reps<a>-<b>.csv` | por réplica y estimador: LL, convergencia, segundos, `mismo_optimo` (cuántos arranques llegan al mismo punto), estadísticas de mercado verdaderas y estimadas |
| `precios.csv`, `distribucion.csv`, `ccps.csv` | de la réplica `rep_reporte`: P(j, a, s), q y CCPs (keep, purge, trade, repair, scrap) verdaderos y con θ̂ |
| `panel_rep<k>.csv.gz` | con `--guardar_panel` |

Columnas completas en `reporte_estimacion.md`, sección 4.

### 4.4 Ejercicio B: Gillingham sobre sus propios datos

Desde `claude/scripts/gillingham/`.  Es mucho más barato (76 estados con a_max = 25).

```bash
cd claude/scripts/gillingham
python -u estimar.py --smoke                                     # humo
TAG_GILL=gill_propios_A25
python -u estimar.py --reps 0:100 --a_max 25 --types low_couple_poor,low_single_poor \
                     --N 20000 --K 10 --tag $TAG_GILL -v
```

`--K 10` = 10 años por hogar, como el paper (modelo_fin usa K = 2 por año × 13 años; las
observaciones no son comparables una a una).  Mismas opciones `--fix`, `--n_starts`,
`--infos parcial,completa`.  Salidas iguales a las de modelo_fin, sin s.

### 4.5 Ejercicio C: el cruce (Gillingham sobre datos con reparación)

Necesita el panel de modelo_fin (paso 1 de 4.3 con `--guardar_panel`).  `--a_max` y
`--types` **tienen que coincidir** con la corrida de modelo_fin.

**Una réplica** (el panel 0):

```bash
cd claude/scripts/gillingham
EST=../../output/estimaciones
TAG_CRUCE=gill_cruce_A25
python -u estimar.py --panel $EST/modelo_fin/$TAG_MF/panel_rep0.csv.gz --reps 0:1 \
                     --a_max 25 --types low_couple_poor,low_single_poor --tag $TAG_CRUCE -v
```

**Monte Carlo del cruce** (necesita los paneles de todas las réplicas): una llamada por
panel, todas con el mismo tag, con `--reps k:k+1` para que la fila lleve la réplica k.
Cada llamada reescribe `precios.csv`/`ccps.csv`/`distribucion.csv`, así que se recorre de
la última a la 0 para que esos archivos queden con el panel 0:

```bash
for k in $(seq 99 -1 0); do
  python -u estimar.py --panel $EST/modelo_fin/$TAG_MF/panel_rep$k.csv.gz --reps $k:$((k+1)) \
                       --a_max 25 --types low_couple_poor,low_single_poor --tag $TAG_CRUCE
done
```

Todas las filas van a `parametros_reps0-0.csv` y `resumen_reps0-0.csv` (el nombre es fijo
con `--panel`; la columna `rep` distingue las réplicas).  La "verdad" contra la que se
compara son los parámetros de Gillingham/`PAPER_TYPES`, que son los mismos que usa
modelo_fin.

### 4.6 Los Monte Carlo viejos (solo para reproducir resultados anteriores)

- `claude/scripts/modelo_fin/main.py`: MC de la **primera etapa** de Hu & Xin (v1.0;
  recupera Pr(reparar | h) y la transición de s, no la estimación estructural).
  `python main.py --smoke`; `python main.py --solve-only ...` y luego
  `python main.py --reps 0:10 ...`.  Escribe en `claude/scripts/modelo_fin/output/`.
- `claude/scripts/gillingham/main.py`: MC de la réplica de v1.3-v1.4.
  `python main.py --reps 0:50 -v`, `python main.py --summarize`.  Escribe en
  `claude/output/montecarlo/gillingham/` (ahí están los resultados de esas versiones).

---

## 5. Resultados y gráficas

### 5.1 Generarlas

`claude/scripts/analisis/main.py` lee las carpetas de los ejercicios A, B y C y escribe
gráficas y tablas.  Cualquier combinación sirve (solo `--mf`, solo `--gill`, ...); `--gill`
se puede repetir.

```bash
cd claude/scripts/analisis
EST=../../output/estimaciones
python -u main.py --mf   $EST/modelo_fin/$TAG_MF \
                  --gill $EST/gillingham/$TAG_CRUCE \
                  --gill $EST/gillingham/$TAG_GILL \
                  --nombre tesis_v1 --todas
```

| opción | para qué |
|---|---|
| `--nombre` | carpeta de salida: `claude/output/analisis/<nombre>/` |
| `--todas` | además las tablas con todos los parámetros (u0, u1 por marca y tipo, dinámica de s) |
| `--regimen t` | año de R para las gráficas de precios (default: el central, t = 6 con T = 13) |
| `--rep k` | réplica para las tablas de parámetros (default: la primera) |

Es CPU y rápido: se puede correr en la supercomputadora o en la laptop (copiando antes
`claude/output/estimaciones/`, ver 6.8).  Se puede correr **mientras el MC sigue**: toma
las réplicas que ya estén escritas.

### 5.2 Qué sale y cómo leerlo

En `claude/output/analisis/<nombre>/graficas/` (cada una en PNG y PDF):

| archivo | qué muestra | qué buscar |
|---|---|---|
| `mc_sesgo` | cajas de estimado − verdad por parámetro y estimador, sobre réplicas | **la gráfica central**: D1 centrada en 0 como D0; Gillingham sobre datos con reparación desplazada |
| `precios_edad` | P(a) (media sobre s ponderada por q) de modelo_fin, con las de Gillingham encima | si θ̂ reproduce los precios |
| `precios_3d_marca<j>` | superficie P(a, s) por fuente | forma de los precios en s |
| `distribucion_edad` | q por edad y marca, todas las fuentes | |
| `distribucion_s` | mapa de calor q(a, s) (eje en ℓ = logit(s) con log-odds) | dónde está la masa en el grid |
| `sin_coche` | fracción de hogares sin coche | |
| `ccps_edad` | Pr(reparar) y Pr(keep) por edad | |

En `.../tablas/` (cada una en CSV, LaTeX booktabs para la tesis, y Markdown para leer):

| archivo | qué tiene |
|---|---|
| `tabla_parametros_principal` | tipo regresión: estimado (se), una columna por estimador, con la verdad |
| `tabla_sesgo_principal` | estimado − verdad con [z]: la prueba que importa en una simulación |
| `tabla_mc` (+ `tabla_mc_numerica.csv`) | sobre réplicas: media, sesgo, RMSE, se medio, sd del MC, cobertura 95% |
| `tabla_mercado` | sin coche, tasa de reparación, accidentes, edad media, precio medio, chatarreo, RMSE de P |
| `..._completa` | con `--todas` |

Colores fijos: verdad negro, D0 azul, D1 naranja, Gillingham sobre sus datos
aqua/violeta, Gillingham sobre datos con reparación rosa/amarillo.

### 5.3 Verlas

- Laptop: abrir los PNG desde VS Code / Positron o el explorador; los `.md` de tablas se
  leen en el editor; los `.tex` se incluyen en la tesis con `\input{...}`.
- Supercomputadora por ssh: no hay pantalla; traer la carpeta a la laptop (6.8) o, si se
  usa VS Code con Remote-SSH, abrir los PNG ahí mismo.

### 5.4 Leer los CSV a mano (pandas)

```python
import glob, pandas as pd
d = "claude/output/estimaciones/modelo_fin/tesis_A25_S100_T13_N20000_K2"
par = pd.concat(map(pd.read_csv, glob.glob(f"{d}/parametros_reps*.csv")))
par = par[par["mejor"]]                                   # el arranque elegido
par.groupby(["estimador", "parametro"]).apply(lambda x: (x.estimado - x.verdad).mean())
res = pd.concat(map(pd.read_csv, glob.glob(f"{d}/resumen_reps*.csv")))
res.groupby("estimador")[["convergio", "segundos"]].mean()
```

---

## 6. Supercomputadora (2 × Quadro GV100), paso a paso

### 6.0 La forma corta: `correr_gpu.sh`

`claude/scripts/correr_gpu.sh` hace las fases 6.4-6.7 de abajo con un comando cada una.
Lo de las secciones 6.1-6.8 es lo mismo escrito a mano (sirve para entender qué hace el
script o para correr algo suelto).

**Una sola vez:** preparar el entorno (6.2, pasos 1-3).  El script activa solo el
entorno virtual si está en `~/venvs/tesis` (si está en otro lado: `VENV=/ruta/al/venv`).

**Cada vez:**

```bash
ssh usuario@super                       # entrar a la máquina
cd ~/tesis_final_1 && git pull          # <- donde esté el clone; traer la última versión
tmux new -s tesis                       # sesión que sobrevive al cerrar ssh

bash claude/scripts/correr_gpu.sh prueba     # 1. pruebas + corrida chica: ¿funciona y cuánto tarda?
bash claude/scripts/correr_gpu.sh todo       # 2. todo lo demás (puede tardar días)
#  Ctrl-b d   -> salir de tmux dejándolo corriendo; cerrar ssh sin miedo
```

Para volver a ver cómo va: `ssh`, luego `tmux attach -t tesis`.  O, sin entrar a tmux:

```bash
bash claude/scripts/correr_gpu.sh estado     # GPUs, procesos vivos, réplicas terminadas
tail -f claude/output/logs/tesis_A25_S100_T13_N20000_K2/mf_1-49.log   # un log en vivo
```

Las fases (se pueden correr una por una en vez de `todo`; cada una espera a que terminen
sus procesos antes de devolver el control):

| fase | qué hace | GPUs | necesita |
|---|---|---|---|
| `prueba` | `jax.devices()`, `tests.py --estructural`, humo de los dos `estimar.py`, y una corrida chica (T = 3, N = 5,000) cuyos tiempos imprime al final | 0 | — |
| `fase1` | GPU 0: verdad + réplica 0 de modelo_fin + panel.  GPU 1: Gillingham sobre sus datos (todas sus réplicas) y `compare.py`.  Al final: cruce de la réplica 0 y un primer análisis (`claude/output/analisis/<tag>_rep0/`) | 0 y 1 | — |
| `fase2` | MC de modelo_fin: réplicas 1..REPS−1 partidas en bloques, mitad en cada GPU | 0 y 1 | `fase1` |
| `cruce` | Gillingham sobre el panel de cada réplica (de la última a la 0) | 0 | `fase2` |
| `analisis` | gráficas y tablas con lo que haya (`claude/output/analisis/<tag>/`) | CPU | cualquier cosa |
| `todo` | `fase1`, `fase2`, `cruce`, `analisis` en orden | | |
| `estado` | `nvidia-smi`, procesos vivos y réplicas terminadas por carpeta | | |

**Cambiar el diseño o la calibración** sin editar el archivo: poner variables delante del
comando.  **Usar las mismas variables en todas las fases de un mismo diseño** (si no,
cada fase busca otra carpeta).

```bash
REPS=50 bash claude/scripts/correr_gpu.sh todo                      # 50 réplicas en vez de 100
PROC_POR_GPU=2 bash claude/scripts/correr_gpu.sh fase2              # 2 procesos por GPU (si GPU-Util es baja)
CALIB=tesis_rep_alta bash claude/scripts/correr_gpu.sh todo         # otra calibración (sección 2.4)
EXTRA="--fix u_s,s_persist" bash claude/scripts/correr_gpu.sh todo  # opciones extra de modelo_fin/estimar.py
```

| variable | default | qué es |
|---|---|---|
| `CALIB`, `A_MAX`, `N_S` | tesis, 25, 100 | calibración y tamaño |
| `T_REG`, `SPREAD` | 13, 0.3 | años de R y su dispersión |
| `N_HOG`, `K_ANIOS` | 20000, 2 | hogares por año, años por hogar (modelo_fin) |
| `TIPOS` | low_couple_poor,low_single_poor | tipos de hogar (iguales en los dos modelos) |
| `REPS` | 100 | réplicas de modelo_fin (0..REPS−1) y del cruce |
| `GILL_REPS`, `GILL_K` | 100, 10 | réplicas y años por hogar de Gillingham sobre sus datos |
| `PROC_POR_GPU` | 1 | procesos de modelo_fin por GPU en `fase2` |
| `EXTRA` | — | opciones adicionales para `modelo_fin/estimar.py` |
| `VENV` | ~/venvs/tesis | entorno virtual |
| `TAG_MF`, `TAG_GILL`, `TAG_CRUCE` | se arman con lo anterior | nombres de las carpetas de salida |

Dónde queda todo: estimaciones en `claude/output/estimaciones/{modelo_fin,gillingham}/<tag>/`,
gráficas y tablas en `claude/output/analisis/<tag>/`, logs en
`claude/output/logs/<TAG_MF>/`.  Para traerlo a la laptop: 6.8.

Si algo falla: el script avisa `OJO: ... revisar los logs` y sigue con lo que puede.  Lo
ya escrito queda guardado; para continuar un bloque caído, ver la regla 6 de 4.2 (se
relanza a mano el rango que falta, con el mismo diseño y tag).

Antes de usarlo en serio:
- **Correr `prueba` primero.**  Todavía no se sabe cuánto tarda una réplica completa; con
  los tiempos que imprime se deciden `REPS` y `PROC_POR_GPU` (6.4).
- **El cruce va en una sola GPU y una réplica tras otra.**  Todas las réplicas del cruce
  escriben en el mismo CSV (`parametros_reps0-0.csv`) y en paralelo podrían mezclar
  filas.  Gillingham es barato, así que no debería ser cuello de botella (no está medido).
- El script no se ha corrido todavía en la supercomputadora (solo se revisó la sintaxis).

Alternativa a tmux: `nohup bash claude/scripts/correr_gpu.sh todo > claude/output/logs/todo.log 2>&1 &`
(sigue corriendo al cerrar ssh; el avance general queda en `todo.log`).

#### Sin tmux: con la ventana de ssh abierta

tmux no es obligatorio.  Lo único que hace es mantener viva la terminal en la
supercomputadora aunque se cierre la conexión; **no usa más la máquina**: las GPUs
trabajan igual con o sin tmux.  Sin tmux, el programa vive mientras viva la ventana de
ssh:

```bash
ssh usuario@super
cd ~/tesis_final_1 && git pull
bash claude/scripts/correr_gpu.sh prueba     # se ve todo en pantalla; al terminar, cerrar
```

El riesgo: si la laptop se suspende, se cae el wifi o se cierra la ventana, el proceso se
detiene.  Lo ya terminado queda guardado (los CSV se escriben réplica por réplica); lo que
iba a la mitad se pierde.  Mientras corre, desactivar la suspensión de la laptop.

Recomendado si no se quiere dejar nada corriendo sin supervisión:
1. `prueba` con el ssh abierto (minutos).  Con los tiempos que imprime, calcular cuánto
   tardaría el Monte Carlo.
2. Correr por fases, cada una con el ssh abierto: `fase1` un día, luego `fase2` con
   pocas réplicas (`REPS=20 bash claude/scripts/correr_gpu.sh fase2`).
3. Para continuar el Monte Carlo después **no** volver a correr `fase2` con un `REPS`
   más grande (empezaría otra vez en la réplica 1).  Lanzar a mano solo el rango que
   falta, con el mismo diseño y tag (regla 6 de 4.2), p. ej. las réplicas 20-39 en
   las dos GPUs:
   ```bash
   cd claude/scripts/modelo_fin
   export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
   mkdir -p ../../output/logs
   DISENO="--calib tesis --a_max 25 --n_s 100 --T 13 --spread 0.3 --N 20000 --K 2 --types low_couple_poor,low_single_poor"
   COMUN="$DISENO --tag tesis_A25_S100_T13_N20000_K2 --rep_reporte 0 --guardar_panel"
   CUDA_VISIBLE_DEVICES=0 python -u estimar.py --reps 20:30 $COMUN > ../../output/logs/mf_20-29.log 2>&1 &
   CUDA_VISIBLE_DEVICES=1 python -u estimar.py --reps 30:40 $COMUN > ../../output/logs/mf_30-39.log 2>&1 &
   wait            # espera a que terminen los dos (no cerrar la ventana)
   ```
4. Si una corrida va a durar días, preguntar a los administradores si se puede dejar un
   trabajo largo; en ese caso, tmux.

#### Ver el uso de las GPUs mientras corre

Si la ventana de ssh está ocupada con la corrida, abrir **otra** ventana de terminal en
la laptop y entrar de nuevo (`ssh usuario@super`); ahí:

```bash
nvidia-smi                 # una foto del momento
watch -n 2 nvidia-smi      # se actualiza cada 2 segundos (Ctrl-c para salir)
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
           --format=csv -l 5       # una línea por GPU cada 5 s, más fácil de leer
nvtop                      # si está instalado: gráfica en vivo tipo "administrador de tareas"
```

Cómo leer `nvidia-smi`:

| columna | qué es | qué esperar |
|---|---|---|
| `GPU-Util` | % del tiempo que la GPU está calculando | alto (> 50%) es bueno; bajo con un solo proceso → probar `PROC_POR_GPU=2` |
| `Memory-Usage` | memoria usada / 32768 MiB | del orden de cientos de MB a pocos GB por proceso; si se acerca a 32 GB, sobran procesos |
| `Pwr:Usage/Cap` | consumo / máximo (250 W) | sube cuando está calculando (en reposo ~24 W) |
| `Temp` | temperatura | normal hasta ~80 °C |
| tabla `Processes` | qué procesos usan cada GPU | debe aparecer `python` en la GPU 0 y/o 1 según la fase |

Los primeros minutos de cada proceso la GPU puede verse casi en 0%: es JAX compilando
(trabaja la CPU).  Para ver la CPU y la memoria RAM: `top` o `htop` (salir con `q`).

### 6.1 Qué máquina es y qué implica

| | |
|---|---|
| GPUs | 2 × NVIDIA Quadro GV100 (arquitectura Volta), **32 GB** cada una |
| driver / CUDA | 535.104.12 / CUDA 12.2 → usar `jax[cuda12]` (no `cuda13`, que ya no soporta Volta) |
| doble precisión | la GV100 tiene FP64 completa (~7 TFLOPS, la mitad de FP32): bien para este código, que corre todo en float64 (`jax_enable_x64`) |
| gestor de colas | por lo que se ve (Xorg, gnome-shell en `nvidia-smi`) es una estación de trabajo, **sin SLURM**: no se usa `sbatch` sino procesos en segundo plano (si sí tiene SLURM, ver 6.9) |

Cómo aprovecharla:

1. **Un proceso de Python usa una sola GPU.**  JAX ve las dos, pero el código no reparte
   un equilibrio entre GPUs.  La forma de usar las dos es **correr procesos
   independientes, cada uno fijado a una GPU** con `CUDA_VISIBLE_DEVICES=0` o `=1`.
2. **La paralelización natural es por réplicas**: el Monte Carlo se parte en bloques
   (`--reps 1:50` en una GPU, `--reps 50:100` en la otra).  Son independientes.
3. **La memoria sobra.**  Un equilibrio de modelo_fin con a_max = 25, n_s = 100 y 2 tipos
   tiene ~21,600 incógnitas y no arma matrices n × n (`--method krylov`); ocupa del orden
   de cientos de MB, no GB.  Por eso caben **varios procesos por GPU**.  Si en
   `nvidia-smi` la columna `GPU-Util` de un proceso se queda baja (< 50%), lanzar un
   segundo proceso en la misma GPU con otro bloque de réplicas.
4. Siempre `XLA_PYTHON_CLIENT_PREALLOCATE=false`: por default JAX reserva el 75% de la
   memoria de la GPU al arrancar y un segundo proceso en la misma GPU fallaría.
5. Los procesos tienen que sobrevivir a que se cierre la sesión de ssh: usar **tmux**
   (recomendado) o `nohup ... &`.

### 6.2 Preparación (una sola vez)

```bash
# 1. Traer el código (el clone ya existe)
cd ~/tesis_final_1                # <- donde esté el clone
git fetch && git checkout modelo-1 && git pull

# 2. Entorno virtual con JAX para CUDA 12
python3 -m venv ~/venvs/tesis
source ~/venvs/tesis/bin/activate
pip install -U pip
pip install -U "jax[cuda12]" jaxopt scipy pandas matplotlib

# 3. Verificar que JAX ve las dos GPUs y usa float64
python -c "import jax; print(jax.devices())"
#   esperado: [CudaDevice(id=0), CudaDevice(id=1)]
#   si sale [CpuDevice(id=0)], JAX no encontró CUDA: revisar el paso 2
CUDA_VISIBLE_DEVICES=1 python -c "import jax; jax.config.update('jax_enable_x64', True); import jax.numpy as jnp; print(jax.devices(), jnp.ones(3).dtype)"
#   esperado: [CudaDevice(id=0)] float64   (con CUDA_VISIBLE_DEVICES=1, la GPU 1 se llama id=0)

# 4. Recursos del resto de la máquina
nproc; free -h; nvidia-smi

# 5. Pruebas en la GPU (minutos)
cd claude/scripts/modelo_fin
export PYTHONIOENCODING=utf-8 XLA_PYTHON_CLIENT_PREALLOCATE=false
CUDA_VISIBLE_DEVICES=0 python -u tests.py --n_s 12 --estructural
CUDA_VISIBLE_DEVICES=0 python -u estimar.py --smoke
cd ../gillingham && CUDA_VISIBLE_DEVICES=0 python -u estimar.py --smoke
```

Si al importar JAX aparece un aviso de que el driver (CUDA 12.2) es más viejo que el
`ptxas` de JAX y que desactiva la compilación en paralelo, **no es un error**: solo
compila un poco más lento.  Si en cambio falla con un error de versión del driver o de
PTX, instalar una versión de JAX algo más vieja (`pip install "jax[cuda12]==<versión>"`,
una que pida CUDA ≤ 12.2) o pedir al administrador que actualice el driver.

### 6.3 Variables de cada sesión

En cada sesión nueva (o al inicio de la ventana de tmux), copiar y pegar:

```bash
cd ~/tesis_final_1                                       # <- donde esté el clone
source ~/venvs/tesis/bin/activate
export PYTHONIOENCODING=utf-8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

RAIZ=$(pwd)
SCR=$RAIZ/claude/scripts
EST=$RAIZ/claude/output/estimaciones
LOGS=$RAIZ/claude/output/logs                            # no se sube a git
mkdir -p $LOGS

DISENO="--calib tesis --a_max 25 --n_s 100 --T 13 --spread 0.3 --N 20000 --K 2 --types low_couple_poor,low_single_poor"
TIPOS=low_couple_poor,low_single_poor
TAG_MF=tesis_A25_S100_T13_N20000_K2
TAG_GILL=gill_propios_A25
TAG_CRUCE=gill_cruce_A25
```

Para otra calibración, cambiar `DISENO` y los tags (p. ej. `--calib tesis_rep_alta` y
`TAG_MF=tesis_rep_alta_A25_...`).

tmux en 30 segundos: `tmux new -s tesis` abre una sesión; `Ctrl-b d` la deja corriendo
y te saca; `tmux attach -t tesis` vuelve a ella; `Ctrl-b c` abre otra ventana,
`Ctrl-b n` pasa a la siguiente.

### 6.4 Fase 0: medir tiempos (obligatoria antes del MC grande)

Todavía no se sabe cuánto tarda una réplica a tamaño completo.  Primero una corrida
chica, y luego una réplica real:

```bash
cd $SCR/modelo_fin
# chica: 3 años, 5,000 hogares (tag aparte para no mezclar)
CUDA_VISIBLE_DEVICES=0 python -u estimar.py --reps 0:1 --calib tesis --a_max 25 --n_s 100 \
    --T 3 --N 5000 --K 2 --types $TIPOS --tag prueba_T3 -v 2>&1 | tee $LOGS/prueba_T3.log
```

Mirar en el log (y en `resumen_reps0-0.csv`, columna `segundos`) cuánto tardan la
verdad y cada estimador.  El costo crece más o menos con T (cada evaluación resuelve un
equilibrio por año), así que la réplica completa (T = 13) tarda del orden de 4 veces
más que esta.  Con eso se decide cuántas réplicas y cuántos procesos por GPU.

### 6.5 Fase 1: réplica 0 de modelo_fin + Gillingham (las dos GPUs a la vez)

```bash
# GPU 0: verdad + réplica 0 de modelo_fin (D0 y D1) + panel
cd $SCR/modelo_fin
CUDA_VISIBLE_DEVICES=0 nohup python -u estimar.py --reps 0:1 $DISENO --tag $TAG_MF \
    --guardar_panel -v > $LOGS/mf_rep0.log 2>&1 &

# GPU 1: Gillingham sobre sus datos, MC completo (es barato)
cd $SCR/gillingham
CUDA_VISIBLE_DEVICES=1 nohup python -u estimar.py --reps 0:100 --a_max 25 --types $TIPOS \
    --N 20000 --K 10 --tag $TAG_GILL -v > $LOGS/gill_propios.log 2>&1 &

# GPU 1 también: comparación de equilibrios G25 vs M25 (un tipo, n_s = 100)
cd $SCR/comparacion
CUDA_VISIBLE_DEVICES=1 nohup python -u compare.py --n_s 100 > $LOGS/compare.log 2>&1 &
```

Esperar a que termine `mf_rep0` (el log termina con `listo: ...`; o
`ls $EST/modelo_fin/$TAG_MF/` muestra `parametros_reps0-0.csv`).  Lo que importa para
seguir es que exista `$EST/modelo_fin/$TAG_MF/verdad.npz`.

En cuanto termine, ya se puede correr el cruce de una réplica y un primer análisis:

```bash
cd $SCR/gillingham
CUDA_VISIBLE_DEVICES=1 python -u estimar.py --panel $EST/modelo_fin/$TAG_MF/panel_rep0.csv.gz \
    --reps 0:1 --a_max 25 --types $TIPOS --tag $TAG_CRUCE -v > $LOGS/cruce_rep0.log 2>&1

cd $SCR/analisis
python -u main.py --mf $EST/modelo_fin/$TAG_MF --gill $EST/gillingham/$TAG_CRUCE \
    --gill $EST/gillingham/$TAG_GILL --nombre ${TAG_MF}_rep0 --todas
```

### 6.6 Fase 2: Monte Carlo de modelo_fin repartido en las dos GPUs

Ejemplo con 100 réplicas (1-99, la 0 ya está), un proceso por GPU:

```bash
cd $SCR/modelo_fin
COMUN="$DISENO --tag $TAG_MF --rep_reporte 0 --guardar_panel"
CUDA_VISIBLE_DEVICES=0 nohup python -u estimar.py --reps 1:50   $COMUN > $LOGS/mf_1-49.log  2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u estimar.py --reps 50:100 $COMUN > $LOGS/mf_50-99.log 2>&1 &
```

Si `nvidia-smi` muestra `GPU-Util` baja con un proceso por GPU, usar dos por GPU (cuatro
bloques):

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u estimar.py --reps 1:25   $COMUN > $LOGS/mf_1-24.log  2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u estimar.py --reps 25:50  $COMUN > $LOGS/mf_25-49.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u estimar.py --reps 50:75  $COMUN > $LOGS/mf_50-74.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u estimar.py --reps 75:100 $COMUN > $LOGS/mf_75-99.log 2>&1 &
```

Más procesos que eso no suele ayudar: compiten por la misma GPU y por los núcleos de CPU
(`nproc`).  Lo que sí ayuda es que ningún bloque sea mucho más largo que los otros.

Nota: la compilación de JAX se repite en cada proceso (los primeros minutos de cada log);
por eso conviene pocos procesos con bloques largos en vez de muchos con bloques cortos.

### 6.7 Fase 3: cruce del MC y análisis final

Cuando terminen los bloques de la fase 2 (o en paralelo, en la GPU que esté libre):

```bash
cd $SCR/gillingham
for k in $(seq 99 -1 0); do
  CUDA_VISIBLE_DEVICES=1 python -u estimar.py --panel $EST/modelo_fin/$TAG_MF/panel_rep$k.csv.gz \
      --reps $k:$((k+1)) --a_max 25 --types $TIPOS --tag $TAG_CRUCE
done > $LOGS/cruce_mc.log 2>&1

cd $SCR/analisis
python -u main.py --mf $EST/modelo_fin/$TAG_MF --gill $EST/gillingham/$TAG_CRUCE \
    --gill $EST/gillingham/$TAG_GILL --nombre $TAG_MF --todas
```

(El cruce de la réplica 0 ya se hizo en la fase 1; repetirlo no duplica filas: el análisis
se queda con la última.)

### 6.8 Vigilar y traer los resultados

```bash
nvidia-smi                       # una foto; `watch -n 5 nvidia-smi` para verla en vivo
tail -f $LOGS/mf_1-49.log        # avance de un bloque (Ctrl-c para salir)
jobs; ps aux | grep estimar.py   # qué procesos siguen vivos
ls $EST/modelo_fin/$TAG_MF/      # qué archivos hay
```

Para matar un proceso: `kill <PID>` (el PID sale en `ps aux`).  Lo hecho hasta la última
réplica completa queda guardado (regla 6 de 4.2).

Traer los resultados a la laptop, dos opciones:

- **Por git** (las estimaciones son CSV chicos; los paneles y logs están en
  `claude/.gitignore`):
  ```bash
  # en la supercomputadora
  git add claude/output/estimaciones claude/output/analisis
  git commit -m "resultados MC tesis_A25 (v1.x)"
  git push
  # en la laptop
  git pull
  ```
- **Por scp** (desde la laptop, en la raíz del repo):
  ```bash
  scp -r usuario@super:~/tesis_final_1/claude/output/estimaciones/ claude/output/
  scp -r usuario@super:~/tesis_final_1/claude/output/analisis/    claude/output/
  ```

### 6.9 Si la máquina sí tiene SLURM

`claude/scripts/slurm_estimar.sh` hace las fases 1-3 con `sbatch` (pasos `mf`, `mf_mc`,
`gill`, `cruce`, `analisis`; instrucciones al inicio del archivo).  Antes: ajustar
`--gres=gpu:1`, los módulos y el entorno al inicio del archivo.  Dos diferencias con este
manual: el paso `cruce` solo corre el panel 0, y `mf_mc` no guarda los paneles (agregar
`--guardar_panel` si se quiere el MC del cruce).

---

## 7. Problemas comunes

| síntoma | causa y arreglo |
|---|---|
| `ModuleNotFoundError: params` (o se importa el módulo equivocado) | no se corrió desde la carpeta del script (sección 0) |
| la primera llamada tarda mucho | compilación de JAX (cada proceso compila una vez) |
| log vacío en segundo plano | falta `python -u` |
| `UnicodeEncodeError` | falta `PYTHONIOENCODING=utf-8` |
| `jax.devices()` solo muestra CPU | se instaló `jax` sin `[cuda12]`, o el entorno virtual no está activado |
| `RESOURCE_EXHAUSTED` / memoria agotada en GPU | falta `XLA_PYTHON_CLIENT_PREALLOCATE=false` con varios procesos por GPU, o se usó `--method dense` a tamaño completo |
| dos procesos en la misma GPU sin querer | falta `CUDA_VISIBLE_DEVICES` |
| `el equilibrio verdadero ... no converge` | calibración extrema; probar con `solve(..., verbose=True)` a n_s chico (sección 2.2) |
| cambié mu (o u0, u1, tc_buy) con `replace` y no pasó nada | salen de `PAPER_TYPES`, no de `Params` (2.5) |
| `repair_price debe ser J x (a_max-1)` | se cambió a_max sin rehacer R: usar `repair_prices(a_max, ...)` (2.4) |
| se = NaN y `cond_B` enorme | una dirección no identificada (pocos datos o un solo tipo de hogar). Más datos, dos tipos con mu distinta o `--fix` |
| `mismo_optimo` < arranques | algún arranque terminó en otro punto: revisar `parametros_*.csv` (están todos los arranques; `mejor` marca el elegido) |
| `precios.csv` no es de la réplica 0 | algún bloque se corrió sin `--rep_reporte 0` (4.2, regla 4) |
| casi todos los hogares sin coche | a_max muy chico (con a_max = 4 los coches casi no valen) |
| precios negativos en coches viejos | chatarreo apagado (`scrap = False`) con a_max grande; "tesis" lo trae encendido |
| .npz de regímenes viejo da error | se guardaron antes de v1.6; regenerar con `main.py --solve-only` |

---

## 8. Dónde está cada documento

| documento (`claude/docs/`) | contenido |
|---|---|
| `manual.md` | esto |
| `calibracion.md` | la calibración "tesis": valores, fuentes y equilibrio resultante |
| `codigo.md` | todos los objetos del código, con diagramas (sección 11: log-odds) |
| `reporte_estimacion.md` | estimadores, columnas de cada CSV, verificación, pendientes |
| `verosimilitud_estructural.md` | la matemática de la verosimilitud |
| `newton_krylov.md` | el solver |
| `modelo_fin.md`, `gillingham.md` | modelos y resultados |
| `datos_ideales.md` | diseños D0-D4 del MC |
| `chatarreo_endogeno.md`, `info_asimetrica.md` | extensiones discutidas |
| `version_hist.md` | qué cambió en cada versión |
