# Diseño de los datasets del Monte Carlo

Estado: **propuesta, no implementada.** Sigue la convención de `scripts/main/gen_dataset.py`
(panel largo en pandas y una opción `cell_based` que agrega a celdas).

## Principio

Un solo simulador genera la "verdad" completa. Cada escenario de datos es una **vista**
de esa verdad con menos columnas. Así las diferencias entre escenarios vienen solo de lo
que se observa, no de simulaciones distintas.

    verdad (todo)  ->  ideal (s por coche)  ->  km (kilometraje por coche)  ->  celdas (agregados)

---

## 1. Regímenes: los "años"

- **T regímenes** t = 1..T (base T = 13), cada uno con su precio de reparación:

      R_t(j, a) = R_base(j, a) * exp(zeta_t)

  zeta_t va en una malla fija (no aleatoria) para controlar la dispersión. Niveles a
  probar: chica (±10%), media (±30%) y grande (±60%).
- **Para cada régimen se resuelve su propio equilibrio** (P_t, q_t, CCPs_t). Los agentes
  tratan R_t como permanente (lectura "un equilibrio estacionario por año").
- **Costo:** los equilibrios son deterministas. Se resuelven **una vez por diseño** y se
  guardan. Las 250 réplicas solo cambian la semilla de la simulación, no los equilibrios.

## 2. Simulación: un panel corto por régimen

Hu & Xin con r binaria necesita parejas (s_t, s_{t+1}) y al menos dos valores de R. Por
eso la base es:

- En cada régimen t se sacan **N hogares de la distribución estacionaria q_t** y se
  simulan **K periodos** con las matrices de ese régimen. Base: K = 2, una transición
  por hogar. Para el escenario km: K = 6 (tres ciclos de inspección).
- Como los hogares vienen de q_t y se simulan con M_t, **los datos son exactamente el
  equilibrio estacionario del régimen t**. No hay transiciones entre regímenes que
  contaminen.
- **Variante de robustez:** un panel largo que cruza regímenes (el mismo hogar vive el
  cambio de R_t a R_{t+1}). Mide el sesgo de suponer estacionariedad cuando el mercado
  está en transición.

Secuencia dentro de un periodo (igual que el modelo):

1. Inicio en x = (j, a, s), terminal o sin coche.
2. Etapa 1: keep / purge / trade. Si trade, se saca h de `buy`.
3. Etapa 2: se saca r con Pr(repair | h).
4. Accidente con prob s_h (o terminal si d = a_max - 1). Si sobrevive, s' ~ F_r.

**Identidad del coche.** Como en Gillingham, el mercado es anónimo: quien compra recibe un
coche sacado de `buy`, no el coche concreto que vendió otro hogar. La unidad del panel es
el hogar, y el coche se sigue mientras el hogar lo tenga (`id_coche` cambia al comprar).

## 3. Tablas

### `verdad` (una fila por hogar × régimen × periodo)

| columna | descripción |
|---|---|
| `id_hogar`, `t`, `k` | hogar, régimen, periodo dentro del régimen |
| `id_coche` | coche que maneja este periodo (cambia al comprar) |
| `estado` | `activo` / `terminal` / `sin_coche` al inicio |
| `j`, `a`, `s` | coche al inicio del periodo (s en el grid) |
| `decision` | `keep` / `purge` / `trade` / `stay_none` |
| `j_h`, `d_h`, `s_h`, `nuevo` | tenencia h después de comerciar (para keep: h = x) |
| `r` | **reparó (oculta en todo escenario salvo el oráculo)** |
| `p_rep`, `p_keep`, `p_trade` | CCPs verdaderas en ese estado (para evaluar) |
| `accidente` | se descompuso este periodo (0/1) |
| `s_next` | s al inicio del periodo siguiente (NaN si accidente) |
| `km` | kilometraje del periodo (solo si el modelo tiene manejo, `idea_km.md`) |
| `P_h` | precio de equilibrio de h (no observado en datos tipo Gillingham) |

### `regimenes` (una fila por régimen)

`t`, `zeta_t`, R_t(j, a), resumen del equilibrio (P_t, masa sin coche, Pr(repair)
promedio). Los arreglos completos (P_t, q_t, CCPs_t) se guardan aparte en `.npz`.

## 4. Escenarios (vistas de `verdad`)

### A. Oráculo: se observa r

Todas las columnas. Es la cota superior: el estimador que no tiene que lidiar con la
mezcla.

### B. Ideal: s por coche, r oculta

Sin `r`, `p_*` ni `P_h`. El estimador usa las parejas (`s_h`, `s_next`) por (j, d,
régimen), juntando keepers y compradores de usados (`estructura_decision.md`). Las filas
con accidente aportan a la tasa de accidentes pero no tienen `s_next`.

### C. Kilometraje: km por coche y por ciclo

Sin `s`, `s_h`, `s_next`, `r`. Se agrega el km por ciclo de inspección (2 periodos) y solo
para edades >= 4. Estimador: Hu & Xin sobre km, solo con coches sin cambio de dueño en el
ciclo (`idea_km.md`).

### D. Celdas: como los datos daneses (`cell_based=True`)

Una fila por (j, a, t) (y por tipo de hogar, si hay heterogeneidad):

| columna | descripción |
|---|---|
| `n` | coches activos en la celda |
| `n_keep`, `n_trade`, `n_purge` | decisiones |
| `n_scrap` | salidas (accidentes, terminal; con inspección: reprobados) |
| `n_buy_j_d` | compras por tipo y edad del coche comprado |

Sin s ni km. Estimador: el modelo con s latente (curvatura del logit o Remark 5).

## 5. Tamaños

- **N hogares por régimen:** 5k, 20k, 100k (Gillingham tiene cerca de 3 millones de
  hogares-año por tipo; 100k es conservador).
- **Celdas finas:** con J = 3, a_max = 7 y n_s = 100 hay 1,800 celdas (j, a, s). Con N =
  20k quedan unas 10 observaciones por celda y régimen. Por eso el estimador de B debe
  **suavizar en s** (sieve como en la sec. 4 de Hu & Xin, o la forma lineal de m), no ir
  celda por celda.
- **Réplicas:** 250 (`mc_replic`). Con 13 regímenes × 100k hogares × 2 periodos son 2.6
  millones de filas por réplica, unos 200 MB en memoria. Conviene generar, estimar y
  guardar solo los estimadores de cada réplica, no los datos.

## 6. Diseño factorial sugerido

| factor | niveles |
|---|---|
| escenario de datos | oráculo, ideal, km, celdas |
| N | 5k, 20k, 100k |
| dispersión de R_t | ±10%, ±30%, ±60% |
| T | 2, 5, 13 |
| sigma_repair | 0.5, 1 |

No hace falta correr todo el producto: la base es (ideal, 20k, ±30%, 13, 1) y luego se
mueve un factor a la vez.

## 7. Implementación (cuando toque)

    claude/scripts/gen_dataset_claude.py
      solve_regimes(g, zetas)          -> lista de (EV_t, P_t, ccps_t, q_t); se guarda en .npz
      simulate_panel(regimes, g, N, K, seed)  -> DataFrame `verdad`
      view(df, escenario)              -> DataFrame del escenario
      to_cells(df)                     -> DataFrame de celdas (como cell_based=True)

- La simulación se vectoriza con `jax.random` sobre hogares: muestrear con las CCPs y
  las transiciones de la tenencia.
- **Requisito previo:** el solver de equilibrio ED(P) = 0 (tuyo). Mientras tanto se
  puede probar todo con P = P0 fijo (equilibrio parcial).
