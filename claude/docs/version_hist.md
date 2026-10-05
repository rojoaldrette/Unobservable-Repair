# Historial de versiones

Formato: por versión, **cambios** (qué se hizo y dónde) y **pendientes** (qué quedó
abierto). Lo más reciente va arriba.

---

## v1.1: réplica de Gillingham (2026-10-04, sin commit)

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
1. **Pasar a modelo_fin la lectura corregida de la Tabla 5.** Hoy usa tc_sell = 0.3454
   y tc_sell_inspect = 0.9106. Decidir antes del Monte Carlo.
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
