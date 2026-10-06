# Manual de uso

Punto de entrada para usar el código de `claude/scripts/`. Para entender cómo está hecho
por dentro, ver `codigo.md`; para el detalle de la estimación y sus salidas,
`reporte_estimacion.md`; para la historia de versiones, `version_hist.md`.

---

## 1. Instalación

Python 3.11 o más reciente con:

```
pip install jax jaxopt scipy pandas matplotlib            # CPU (laptop)
pip install "jax[cuda12]" jaxopt scipy pandas matplotlib  # GPU (supercomputadora)
```

Siempre:
- correr desde la carpeta del script (los imports son locales);
- con `PYTHONIOENCODING=utf-8` (hay acentos en los mensajes);
- con `python -u` si corre en segundo plano (si no, el log queda vacío hasta el final).

---

## 2. Mapa rápido: ¿qué quiero hacer?

| quiero... | comando (desde `claude/scripts/`) | sale en |
|---|---|---|
| comprobar que todo funciona | `cd modelo_fin && python -u tests.py --n_s 12 --estructural` | pantalla |
| resolver un equilibrio de modelo_fin | ver sección 3 (Python) | — |
| comparar equilibrios de modelo_fin y Gillingham | `cd comparacion && python compare.py --n_s 24` | `comparacion/output/` |
| estimar modelo_fin con r observada y no observada | `cd modelo_fin && python -u estimar.py --reps 0:1 --guardar_panel -v` | `claude/output/estimaciones/modelo_fin/<tag>/` |
| Monte Carlo de modelo_fin | `python -u estimar.py --reps 0:10` (y más bloques) | misma carpeta |
| estimar Gillingham sobre sus datos | `cd gillingham && python -u estimar.py --reps 0:1 --a_max 7 -v` | `claude/output/estimaciones/gillingham/<tag>/` |
| estimar Gillingham sobre datos con reparación (sesgo) | `cd gillingham && python -u estimar.py --panel <panel_rep0.csv.gz> --a_max 7 -v` | misma carpeta |
| gráficas y tablas | `cd analisis && python main.py --mf <carpeta> --gill <carpeta> --nombre v1 --todas` | `claude/output/analisis/v1/` |
| correr todo en la supercomputadora | `sbatch --export=ALL,STEP=<paso> slurm_estimar.sh` (sección 6) | las mismas carpetas |
| primera etapa de Hu & Xin sola (MC viejo, v1.0) | `cd modelo_fin && python main.py --smoke` | `modelo_fin/output/` |
| MC de la réplica de Gillingham (v1.3-v1.4) | `cd gillingham && python main.py --reps 0:50 -v` | `claude/output/montecarlo/gillingham/` |

Todo comando acepta `--help`.

---

## 3. Usar el modelo desde Python

```python
import dataclasses
from params import Params, Types
from estimar import calibracion            # calibraciones con nombre
from theta import economy
from equilibrium import solve, split_z, equilibrium_objects

g = calibracion("tesis_v0", a_max=7, n_s=100)        # o Params() / dataclasses.replace(...)
eco = economy(g, Types())                             # dos tipos de hogar, mitad y mitad
z, convergio, it = solve(eco, method="krylov", verbose=True)
EVs, P = split_z(z, eco)                              # P: (J, a_max-1, n_s)
objs = equilibrium_objects(z, eco)                    # por tipo: q, keep, purge, trade, buy, repair
```

- Cambiar un parámetro: `dataclasses.replace(g, mu=0.12)` (Params es inmutable).
- Otro año de precios de reparación: `eco.with_repair_price(R)`, con R de forma (J, a_max−1).
- Un solo tipo: `Types(("low_couple_poor",), (1.0,))`.
- La API vieja con g estático sigue funcionando (`ED.solve_equilibrium(g)`), pero
  recompila con cada cambio de parámetro y tiene un solo tipo.

---

## 4. Estimar (`modelo_fin/estimar.py`)

Qué hace: resuelve los equilibrios verdaderos de T años (R_t = R·exp(zeta_t)), simula el
panel, estima con D0 (`oraculo`, r observada) y D1 (`hx`, r no observada) y escribe los
CSV.

Opciones que más se usan:

| opción | default | para qué |
|---|---|---|
| `--reps a:b` | 0:1 | réplicas (semillas) a:b−1 |
| `--calib` | tesis_v0 | `defaults`, `gill_s`, `tesis_v0` |
| `--a_max`, `--n_s` | 7, 100 | tamaño del modelo |
| `--T`, `--spread` | 13, 0.3 | años de R y su dispersión |
| `--N`, `--K` | 20000, 2 | hogares por año y años por hogar |
| `--infos` | oraculo,hx | qué estimadores |
| `--fix` | — | dejar parámetros fijos en la verdad (`--fix u_s,s_persist`) |
| `--n_starts` | 2 | arranques perturbados además de la verdad |
| `--method` | krylov | `dense` solo para tamaños chicos |
| `--guardar_panel` | no | guarda el panel (lo necesita el cruce con Gillingham) |
| `--tag` | automático | nombre de la carpeta de salida |
| `--smoke` | — | tamaño de juguete para probar (minutos) |

**Reanudar o ampliar un Monte Carlo:** correr otro bloque con el mismo diseño (mismo
`--tag`). Reutiliza `verdad.npz` y escribe `parametros_reps<a>-<b>.csv` aparte; el
análisis junta todos los bloques.

**Gillingham (`gillingham/estimar.py`):** mismas ideas. `--infos parcial,completa`
(`parcial` = la del paper). Con `--panel`, a_max y `--types` tienen que coincidir con la
corrida de modelo_fin que generó el panel.

---

## 5. Analizar (`analisis/main.py`)

```
python main.py --mf  ../../output/estimaciones/modelo_fin/<tag> \
               --gill ../../output/estimaciones/gillingham/<tag_cruce> \
               --gill ../../output/estimaciones/gillingham/<tag_propios> \
               --nombre v1 [--regimen 6] [--rep 0] [--todas]
```

Cualquier combinación de corridas sirve (solo `--mf`, solo `--gill`, ...). Las gráficas y
tablas que salen están listadas en `reporte_estimacion.md`, sección 5. Los CSV de
`claude/output/estimaciones/` están en formato largo para trabajarlos con pandas
(columnas en `reporte_estimacion.md`, sección 4).

---

## 6. Supercomputadora

`claude/scripts/slurm_estimar.sh`. Antes de usarlo: ajustar partición, `--gres`, módulos y
entorno al inicio del archivo, y el diseño (A_MAX, N_S, T_REG, ...).

```
cd claude/scripts
sbatch --export=ALL,STEP=mf slurm_estimar.sh                  # 1. modelo_fin, réplica 0 + panel
sbatch --array=1-9 --export=ALL,STEP=mf_mc slurm_estimar.sh   # 2. MC, réplicas 10-99 (después de 1)
sbatch --export=ALL,STEP=gill slurm_estimar.sh                # 3. Gillingham sobre sus datos
sbatch --export=ALL,STEP=cruce slurm_estimar.sh               # 4. Gillingham sobre el panel (después de 1)
sbatch --export=ALL,STEP=analisis slurm_estimar.sh            # 5. gráficas y tablas (después de 1, 3, 4)
```

Para encadenar: `sbatch --dependency=afterok:<jobid> ...`. Logs en `claude/scripts/logs/`.

Recomendado antes del MC: una corrida chica para medir tiempos (editar el diseño a
`T_REG=3`, `N_HOG=5000` y lanzar solo `mf`).

---

## 7. Problemas comunes

| síntoma | causa y arreglo |
|---|---|
| la primera llamada tarda mucho | compilación de JAX; las siguientes son rápidas. Con la API vieja (g estático) recompila en cada cambio de parámetro: usar `theta`/`equilibrium` |
| `ModuleNotFoundError: params` | no se corrió desde la carpeta del script |
| log vacío en segundo plano | falta `python -u` |
| `UnicodeEncodeError` | falta `PYTHONIOENCODING=utf-8` |
| `el equilibrio verdadero ... no converge` | calibración extrema; revisar con `solve(..., verbose=True)` y empezar con `--method dense` a tamaño chico |
| se = NaN y `cond_B` enorme | una dirección no identificada (pocos datos o parámetro sin variación). Más datos o `--fix` |
| `mismo_optimo` < arranques | algún arranque terminó en otro punto: revisar `parametros_*.csv` (están todos los arranques) |
| casi todos los hogares sin coche | a_max muy chico (con a_max = 4 los coches casi no valen) |
| memoria agotada en GPU | usar `--method krylov` (dense arma matrices de n²) |
| .npz de regímenes viejo da error | se guardaron antes de v1.6 (Q0/Q1); regenerar con `main.py --solve-only` |

---

## 8. Dónde está cada documento

| documento | contenido |
|---|---|
| `manual.md` | esto |
| `codigo.md` | todos los objetos del código, con diagramas |
| `reporte_estimacion.md` | estimadores, salidas (columnas), verificación, pendientes |
| `verosimilitud_estructural.md` | la matemática de la verosimilitud |
| `newton_krylov.md` | el solver |
| `modelo_fin.md`, `gillingham.md` | modelos y resultados |
| `datos_ideales.md` | diseños D0-D4 del MC |
| `chatarreo_endogeno.md`, `info_asimetrica.md` | extensiones discutidas |
| `version_hist.md` | qué cambió en cada versión |
