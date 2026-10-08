# Reporte: Monte Carlo completo (2026-10-08)

50 réplicas (0-49) del diseño de `modelo_fin.md`: 10,000 hogares por régimen, 4 años, 3
regímenes de R (90,000 transiciones por réplica). Corrido en la workstation: corrida base
(commit 691c378) + re-estimación de las 36 estimaciones perdidas con `--solo_faltantes`
(commit 6382556). **Ahora las 150 estimaciones de los diseños y las 100 de Gillingham están
completas.**

- Datos: `claude/niu/output/modelo_tesis/montecarlo/base/`.
- Gráficas y tablas: `claude/niu/output/modelo_tesis/reportes/mc_base/` (`analisis/reporte_mc.py`,
  solo lee CSV; corre en local).
- Réplica 0 en detalle: `reporte_rep0.md`.

## Hallazgos

### 1. El modelo de la tesis no se sesga, en ninguno de los tres diseños

50 réplicas por diseño.

- Sesgo medio < 1.5% del valor verdadero en todos los parámetros salvo Ts (−1.5, −2.4 y
  −2.9% en los diseños 1, 2, 3; t del sesgo −1.0 a −1.7, no significativo).
- Único |t| > 2: Tb (pareja), +0.5-0.6% en los diseños 2 y 3 (t = 2.1 y 2.3; 1.7 en el
  diseño 1). Con 27 parámetros × 3 diseños se esperan algunos |t| > 2 por azar, pero sale
  con el mismo signo en los tres diseños (son los mismos paneles, no son independientes).
  En magnitud es despreciable frente al +19-30% de Gillingham; probablemente sesgo de
  muestra finita del MLE.
- Cobertura del IC 95% entre 0.90 y 1.00 (banda binomial con 50 réplicas: ~0.88-1.00).
- sd entre réplicas / se mediano entre 0.78 y 1.21: los errores estándar BHHH están bien.
- Los parámetros de reparación (κ, σ_η, σ_rep, u_w, δ) se recuperan sin sesgo aun en el
  diseño 3 (sin r ni motivo de salida); no ver r agranda la dispersión, sobre todo la de δ.
- La tasa de reparación implícita sale 0.276, igual a la verdadera.
- Perder información cuesta poco: del diseño 1 al 3 el sesgo de los parámetros comunes
  sube algo (μ −0.1% → −0.5%) y la cobertura baja un poco, sin salir de la banda.

![sesgo](../output/modelo_tesis/reportes/mc_base/fig1_sesgo.png)

### 2. Gillingham sobre los mismos datos está muy sesgado

| | parcial (la del paper) | completa |
|---|---|---|
| μ | −57% | −38% |
| u0 | −29 a −49% | −20 a −33% |
| u1 | +37 a +51% | +26 a +36% |
| Tb | +29% | +19% |
| Tb sin coche | +62 a +108% | +41 a +71% |
| Ts | −248% | −174% |
| Ts inspección | −104% | −73% |
| cobertura | 0 en los 16 parámetros de preferencias y costos | 0 |

Con μ (utilidad marginal del dinero) 40-57% más baja, todo lo que Gillingham expresa en
dinero (u/μ, costos de transacción, disposición a pagar) sale inflado.

![cobertura](../output/modelo_tesis/reportes/mc_base/fig3_cobertura.png)

### 3. Las 36 estimaciones perdidas ya se recuperaron (y no cambian nada)

En la corrida base se perdieron 36/150 estimaciones (12, 11, 13 en los diseños 1-3), todas
con "el equilibrio no converge en el valor inicial de BHHH": BHHH arrancaba desde el
último punto de prueba de L-BFGS, no desde el mejor. Se arregló en `ll_estim.py`
(BHHH arranca en la mejor evaluación, `restore_best`, con rescates desde el mejor y desde
cero) y se re-estimaron con `montecarlo.py --solo_faltantes` (mismo panel, misma semilla).

Resultado de la re-estimación (logs `modelo_tesis/mc_faltantes_*.log`):
- 36/36 convergen (criterio de BHHH g'B⁻¹g/N < 1e-9). Ningún rescate hizo falta
  (`rescate_mejor = rescate_frio = 0`): bastó arrancar BHHH en el mejor punto.
- Todas tienen el mismo patrón: **2 evaluaciones fallidas y ~8 evaluaciones en total**
  (vs ~200 en las demás). Es decir, en estas réplicas el primer paso de L-BFGS cae en un
  θ donde el equilibrio no converge desde la cuerda, L-BFGS se rinde tras dos intentos
  (devolvemos 1e10 con gradiente cero) y quien estima es BHHH solo, desde el valor
  verdadero. Eso es lo que antes tiraba la estimación: el estado guardado era el del
  punto fallido. Como BHHH llega al mismo criterio de convergencia, la estimación es
  válida; solo cambia el camino.
- **No hay selección.** La preocupación del reporte anterior era que las réplicas perdidas
  no fueran al azar (en ellas Gillingham daba μ y u0 0.5-0.7 sd más bajos). Con los
  diseños ya estimados, el estadístico z = (θ̂ − θ)/se de las 36 recuperadas vs las 114
  originales: media +0.03 vs −0.01, sd 0.96 vs 0.98 (todos los parámetros); para μ,
  −0.18 vs −0.09 (72 vs 228 obs.). Sin diferencia relevante. Los números de las secciones
  1-2 (que ya incluyen todo) casi no se movieron respecto a la versión con 114
  estimaciones.

Pendiente menor (no urgente): el primer paso de L-BFGS es demasiado largo en ~1/4 de los
paneles. Si se quiere que L-BFGS haga su trabajo, acotar el paso inicial (p. ej. escalar
el objetivo o usar `maxls`/un primer paso de BHHH). Para el MC no hace falta.

![fallas](../output/modelo_tesis/reportes/mc_base/fig5_fallas.png)

### 4. Costo

~130 s y ~200 evaluaciones por estimación en GPU (las 36 re-estimadas: ~160 s y ~8
evaluaciones, más refactorizaciones porque la cuerda da pasos largos). Re-estimar las 36
tomó ~0.9 h en dos GPUs. Gillingham ~85-105 s por estimación en CPU (compitiendo por CPU con
la corrida de los diseños).

![se](../output/modelo_tesis/reportes/mc_base/fig4_se.png)
![reparación](../output/modelo_tesis/reportes/mc_base/fig2_reparacion.png)
