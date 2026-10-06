# Historial de versiones

Formato: por versión, **cambios** (qué se hizo y dónde) y **pendientes** (qué quedó
abierto). Lo más reciente va arriba.

---

## v1.6: modelo_fin sin matrices densas (2026-10-06)

Fase 1 del plan. Detalle en `modelo_fin.md`, "Sin matrices densas".

### Cambios
- **`transitions.py`:**
  - Q como kernel (J, A-1, S, S);
  - M v y q M por la estructura de Ω (diagonal + rango 1 + una columna);
  - `stationary_q`: q exacta por recursión en la edad, sin resolver sistemas.
  - Las funciones densas quedan solo para pruebas.
- **`bellman.nk_step`:** Newton-Kantorovich con GMRES en vez de `solve` denso.
- **`ED.py`:** q por recursión; el sistema (I − βM) dEV = dT/dP dP con GMRES en vez de LU.
- **`utils.gmres`:** GMRES propio, anidable (el de jax.scipy no se puede anidar).
- **`gen_dataset.py`:** simula con el kernel F; los regímenes guardan F, no Q0/Q1.
- `params.gmres_maxiter` (nuevo).
- `tests.py`: `test_matrix_free`.

### Resultados
- Todo coincide con la versión densa (tag `matrices-densas`): operadores a ~1e-15,
  equilibrio con P a 1e-7. Más rápido: 124 s -> 56 s con n_s = 24.
- **a_max = 25 y n_s = 100 ya caben:** Bellman + q en 9 s en laptop.
- El equilibrio a tamaño completo no se corrió (se deja para la supercomputadora).

### Pendientes
- Fases 2-5 del plan (v1.5).
- Los .npz de regímenes guardados con la versión anterior traen Q0/Q1 en vez de F:
  regenerarlos.

---

## v1.5: plan y documentos de diseño (2026-10-05)

Sin cambios de código. Es el último estado de `modelo_fin` con matrices densas (tag
`matrices-densas`).

### Documentos nuevos
- `info_asimetrica.md`: un precio por (j, a) y s descubierta al comprar. Baja los
  precios de 7,200 a 72, pero agrega las creencias π como otro punto fijo.
- `newton_krylov.md`: Newton con GMRES y productos jvp, sin armar el jacobiano.
- `verosimilitud_estructural.md`: segunda etapa, con la mezcla sobre r adentro de un
  DNFXP; gradiente implícito; qué identifica a cada parámetro.
- `chatarreo_endogeno.md`: por qué el chatarreo puede volver como opción sin romper la
  mezcla de Hu & Xin (ℓ' se mide antes de decidir chatarrear; s(ℓ) da la tasa de
  accidentes), y las pruebas que lo garantizan.
- `datos_ideales.md`: escalera de bases de datos para el MC. Seguros: D0 (r observada)
  y D1 (r no observada, sin precios, como Gillingham + ℓ).

### Decisiones
- s en log-odds: ℓ' = ℓ + acc_age_j − s_repair r + η, ℓ_nuevo = acc_int_j (Tabla 4).
- Mismos parámetros que Gillingham, para medir el efecto de poder reparar.
- Dos tipos de hogar (Couple Poor y Single Poor) para identificar tc_buy y tc_sell.
- R(j, a) bajo / medio / alto por marca (4 / 6.5 / 10 mil DKK en la edad 1, +6% por año).
- Chatarreo endógeno como opción (apagado por defecto).
- Reparación parcialmente observada: fuera de la tesis.

### Plan
1. Quitar las matrices densas (kernel de Q, M·v estructurado, GMRES).
2. Parámetros dinámicos, dos tipos, chatarreo opcional, Newton-Krylov conjunto.
3. Calibración (a_max = 25, log-odds, R).
4. Verosimilitud estructural.
5. Monte Carlo listo para GPU.
6. Corridas en GPU (el usuario).

---

## v1.4: tipos de hogar en la réplica de Gillingham (2026-10-05)

### Cambios
- **Tipos de hogar:**
  - `params.PAPER_TYPES` y `GTypes` (Tablas 7-10);
  - `theta.Economy` (mu, u0, u1, tc_buy, tc_buy_nocar por tipo; Ts, sigma_sell y
    accidentes comunes).
- **Equilibrio, simulación y verosimilitud para T tipos** con P común. Con T = 1
  reproduce v1.3 exactamente.
- `theta.free_spec` ahora recibe th (no g), y `LLEval` recibe las fracciones f.
- MC con 3 diseños (N = 40,000, 50 réplicas). Detalle en `gillingham.md`, "Tipos de hogar".

### Resultados
- **Con dos tipos de mu distinta, tc_buy y tc_sell se identifican sin precios:** se de
  2.69 -> 0.16 y RMSE de P de 13.3 -> 1.1 mil DKK.
- **Con dos tipos de mu casi igual, no:** se de 1.5 y RMSE de 7.8.
- **Conclusión:** lo que identifica es la heterogeneidad en mu. Corrige la
  recomendación de v1.3 ("fijar tc_buy"): eso solo hace falta con un tipo.

### Pendientes nuevos
1. Guardar las estimaciones de todos los arranques para distinguir máximos locales de
   arranques mal pulidos.
2. Decidir si `modelo_fin` tiene tipos de hogar con mu distinta (depende de los datos).

---

## v1.3: estimación DNFXP y Monte Carlo de Gillingham (2026-10-04)

### Cambios
- **Parámetros dinámicos** (`gillingham/theta.py`): `GModel(g, th)` es un pytree, así
  que un θ nuevo no recompila.
  - `T`, `ccps` y `physical_matrix` tienen versiones `*_raw` sin jit.
  - `sell_cost` usa jnp.where.
  - Lo demás no cambia: `tests.py` da los mismos números.
- **`equilibrium.py`:** Newton conjunto sobre (EV, P). 3 s en frío contra 26 s; mismo
  equilibrio a 1e-10.
- **`simulate.py`:** panel de hogares desde q, con chatarreo endógeno y accidentes
  marcados, y agregación a celdas.
- **`loglikelihood.py`:** verosimilitud parcial (apéndice D: sin precios ni
  accidentes) y completa (oráculo).
  - Gradiente por función implícita: coincide con diferencias finitas a 5e-8.
  - Optimización: L-BFGS y luego BHHH.
- **`montecarlo.py`, `main.py`:** MC con varios arranques. Mide sesgo, RMSE, se y
  cobertura, más precios, chatarreo endógeno y accidentes en θ̂.
- **Resultados:** 50 réplicas, a_max = 25, N = 20,000, K = 10, en
  `claude/output/montecarlo/gillingham/`. Detalle en `gillingham.md`.

### Resultados
- **El estimador del paper recupera θ sin sesgo y con cobertura ~95%,** incluso sin
  observar accidentes. La fracción de salidas por chatarreo voluntario (0.42) se
  recupera.
- **Sin precios, tc_buy y tc_sell casi no se identifican por separado.** Solo lo
  rompen p_new y p_scrap.
  - se de 3.8, correlación −1 entre réplicas;
  - RMSE de P de 18 mil DKK, contra 0.9 si se fija tc_buy.
  - **Relevante para modelo_fin.**

### Pendientes nuevos
1. Precios de usados como dato (o tc_buy fijo) en la estimación estructural.
2. Pasar `GModel` + Newton conjunto + gradiente implícito a `modelo_fin` (resuelve el
   pendiente 3 de v1.0: `g` estático en jit).

---

## v1.2: corrección de la Tabla 5 en modelo_fin (2026-10-04)

### Cambios
- `modelo_fin/params.py`: tc_sell 0.3454 -> **0.9106**, tc_sell_inspect 0.9106 ->
  **2.1929** (lectura raw de la Tabla 5, igual que la réplica de Gillingham).
  lambda_s = 0.3454 no se usa porque modelo_fin no tiene chatarreo.
- Documentos que citaban la lectura vieja actualizados (`discusion_repair_o1.md`,
  `inspeccion_zigzag.md`, `pruebas_robustes.md`).

### Pendientes
- Igual que v1.1, salvo el punto 1 (resuelto). Sigue por confirmar la Tabla 5 con la
  versión publicada.
- El Monte Carlo de prueba y las comparaciones de v1.0/v1.1 (fila M7) se corrieron con
  la lectura vieja. M7_gill ya usaba la nueva.

---

## v1.1: réplica de Gillingham (commit 01dea15)

### Cambios
- **Réplica de Gillingham** en `claude/scripts/gillingham/` (mismo layout que modelo_fin,
  sin s). Equilibrio por Newton con jacobiano denso: converge en 5–6 pasos, max|ED| ~
  1e-13, y dT/dEV == βM. Detalle en `gillingham.md`.
- **`claude/scripts/comparacion/compare.py`:** compara G25, G7, M7 y M7_gill.
- **Corrección de la lectura de la Tabla 5** (texto raw): sigma_sell = 0.3454,
  tc_sell = 0.9106 y tc_sell_inspect = 2.1929 (la estimación es −2.1929, leída como
  coeficiente de utilidad). La réplica usa esta lectura.

### Resultados
- **Con s calibrada a la Tabla 4, modelo_fin reproduce a Gillingham (a_max = 7):**
  precios a 1–2 mil DKK y keep a menos de 0.02. La reparación baja los hogares sin
  coche de 0.73 a 0.65.
- **a_max = 7 explica casi todo el exceso de hogares sin coche:** 0.73 contra 0.016
  con a_max = 25, mismos parámetros.

### Pendientes nuevos
1. ~~Pasar a modelo_fin la lectura corregida de la Tabla 5.~~ Hecho en v1.2.
2. Confirmar la Tabla 5 con la versión publicada (JPE 2022) o el código de los autores.
3. Adoptar como defaults la calibración de s de M7_gill (o una similar) y subir a_max.
4. **Gillingham:** u2 y el dummy de edad par no están en las tablas (hoy en 0).
   p_scrap y beta tampoco se reportan.
5. Comparar a a_max = 25 también con modelo_fin (requiere matrices dispersas).

---

## v1.0: "agregar modelo 1" (2026-10-04)

Primera versión completa del modelo con reparación no observada, hasta el Monte Carlo.
Código en `claude/scripts/modelo_fin/`, documentación en `claude/docs/`.

### Cambios

**Modelo**
- **Sin chatarreo endógeno.** Un coche activo solo sale por venta (trade o purge),
  accidente (prob s) o edad terminal. `p_scrap` lo cobra solo el dueño de un coche
  terminal.
- **Costos de transacción de Gillingham.**
  - Comprador: Tb = 6.5944 más 1.7899 si compra desde "sin coche" (Tabla 10).
  - Vendedor: Ts = 0.3454, o 0.9106 en años de inspección (edades pares >= 4) (Tabla 5).
- **Decisión secuencial.** Etapa 1: keep / purge / trade (+ qué h comprar). Etapa 2:
  reparar o no sobre la tenencia h, con su propio shock (`sigma_repair`).
  Pr(repair | h) no depende de cómo se llegó a h (`estructura_decision.md`).
- **Transición de s: opción 2.** Reparar mueve la media de s' y no la probabilidad de
  descomponerse este periodo (`discusion_repair_o1.md`).
- **Nido de s al comprar** (`sigma_s` = 0.1). Quita la dependencia del equilibrio
  respecto al tamaño del grid y los precios absurdos en celdas casi vacías.
- **R(j, a) x5** respecto a la base original, a (15, 12.5, 25) mil DKK. Con
  `sigma_repair` = 1, la reparación "por puro ruido" baja de ~40% a 5–15%.

**Código** (separado por componentes, como el repo de Rust)
- `params`, `utils`, `primitives`, `bellman`, `probabilities`, `transitions`: el modelo.
- **`ED.py`: exceso de demanda y equilibrio.** ED_log = log D − log max(S, piso).
  Tâtonnement diagonal y luego Newton-Krylov (jvp + GMRES + función implícita para EV).
- `gen_dataset.py`: regímenes de R (un equilibrio por "año"), simulación del panel y
  vistas (oráculo, ideal, celdas).
- `loglikelihood.py`: estimadores oráculo / ingenuo / Hu & Xin paramétrico (CCPs
  libres por año o logit en R).
- `montecarlo.py`, `main.py`, `slurm_mc.sh`: Monte Carlo por bloques de réplicas para
  la supercomputadora.
- `tests.py`: pruebas del modelo y del equilibrio.

**Verificaciones**
- Filas de Ω y Q suman 1; M rápida == M densa (~1e-16); q = qM.
- dT/dEV == βM contra autodiff (1e-15).
- J_ED·v (jvp) contra diferencias finitas (5e-5).
- El equilibrio converge de forma cuadrática en n_s = 12 y 24.
- Prueba de humo del Monte Carlo (n_s = 12, T = 2, N = 3,000): hx recupera s_repair =
  0.066 (verdad 0.06); el ingenuo duplica s_sigma.

**Documentos**
`fases_tesis.md`, `estructura_decision.md`, `discusion_repair_o1.md`,
`inspeccion_zigzag.md`, `id_modelo_vsHu.md`, `idea_km.md`, `datasets_montecarlo.md`,
`pruebas_robustes.md`, `modelo_fin.md`.

### Pendientes

**Antes de correr el Monte Carlo en la supercomputadora**
1. **Calibración de s.** Con los defaults, los accidentes son de 3–20% al año
   (Gillingham, Tabla 4: 0.4–2%) y en equilibrio el 92–96% de los hogares queda sin coche
   (Gillingham: cerca de 44%). Con s tipo Tabla 4 (`s_new` = 0.004, `s_const` = 0.002,
   `s_age` = 0.0003, `s_sigma` = 0.003, `s_repair` = 0.004, `s_max` = 0.1) baja a 62%.
   Hay que decidir los defaults. Al cambiar s, revisar también R y `sigma_repair`,
   para que Pr(repair) siga en un rango con información (hoy 5–17%).
2. **a_max = 7** con parámetros de Gillingham calibrados para 25 edades: los coches
   viven 6 años. Probablemente explica el resto del exceso de hogares sin coche. Subir
   a_max implica matrices densas de 7,204 x 7,204 (~415 MB): conviene usar matrices
   dispersas antes.

**Estimación**
3. **Segunda etapa estructural** (mu, sigma_repair, costos de transacción). `g` es
   estático en jit, así que cada valor nuevo de parámetros recompila. Hay que separar
   los parámetros que se estiman en un pytree dinámico.
4. **Estimador para datos por celdas** (tipo daneses, s latente). `to_cells` ya genera
   los datos.
5. **Escenario de kilometraje.** Requiere manejo en el modelo (`idea_km.md`).

**Modelo**
6. **Precios de celdas sin masa.** Se fijan donde D = piso y no tienen significado
   económico. Si en algún equilibrio aparecen precios negativos en celdas con masa,
   decidir el piso (reparación como piso o comprador de chatarra; ver el chat y
   `inspeccion_zigzag.md`).
7. **Inspecciones** (idea A como base, B como extensión): sin implementar.
8. **Por verificar en Gillingham:**
   - si 0.9106 es el nivel o un incremento sobre 0.3454;
   - si el dueño de un coche terminal paga el costo "no car" al comprar.
9. Decidir si se **prohíbe reparar en a_max − 1** (no tiene efecto en la transición).
10. Decidir la lectura de los "años": un equilibrio estacionario por R_t o R como estado
    agregado.

**Siguiente versión**
- **Réplica del modelo de Gillingham** en `claude/scripts/gillingham/` para comparar su
  equilibrio con el de este modelo (con la orden "haz Gillingham").
