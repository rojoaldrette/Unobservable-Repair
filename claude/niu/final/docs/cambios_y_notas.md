# niu/final: cambios respecto a niu/modelo_tesis y notas para la tesis

Creado 2026-11-17. `niu/final/` es una copia limpia (solo `.py`) de `niu/gillingham/` y
`niu/modelo_tesis/` ajustada a `docs/tex/tesis.tex`. Las carpetas viejas quedan intactas
(son las que generaron el MC de octubre). Las salidas nuevas van a `niu/final/output/`.

## 1. Cambios en el código

| Cambio | Archivos | Efecto |
|---|---|---|
| $T_s$ constante en la edad: se quitan `tc_sell_inspect` / `TC_SELL_INSPECT`, `inspect_age_min` e `inspection_years`. | `gillingham/{params,bellman,ll_estim,montecarlo}.py`, `modelo_tesis/{params,bellman,ll_estim}.py` | Sin zig-zag. El modelo de la tesis queda con **26** parámetros libres (antes 27); el de Gillingham, con uno menos. |
| Deriva de $w$: $m=w+\delta_j-\kappa r$ (antes `w + delta - acc_age - kappa r`). $\delta_j=\kappa\bar p=0.075$ para las dos marcas. | `modelo_tesis/{bellman,params,tests}.py` | Con los valores verdaderos, la economía es la misma. $\hat\delta$ cambia de significado: $\hat\delta_{\text{nuevo}}=\hat\delta_{\text{viejo}}-\hat\alpha_1$. |
| Se quita `u2` (edad² en la utilidad, antes fija en 0). | `modelo_tesis/{bellman,params,ll_estim}.py` | Ninguno numérico. Gillingham conserva su `u2 = 0`. |
| $R$ de heavy brown **constante en la edad**: 16.6 mil DKK, el promedio en las edades 1–23 de la recta anterior $10+0.6(a-1)$. Light brown sigue en U: $4+0.1(a-1)+6e^{-(a-1)/4}$. | `modelo_tesis/params.py` | Cambia la economía. Hay que revisar en GPU que haya reparación razonable en heavy brown. |

Sin cambio, solo notación: $T_b$, $T_s$ en utils, restados directamente (como Gillingham, lectura `tc_units = "utils"`).

Pruebas (`python -u tests.py` en cada carpeta, CPU, a_max chico): todas pasan, incluido el
gradiente implícito de los diseños 1–3 y de las verosimilitudes parcial y completa de Gillingham.
En Windows, correr con `PYTHONIOENCODING=utf-8`; si no, falla al imprimir "≈".

2026-11-18: inferencia de la reparación por observación (`modelo_tesis/inferencia.py`, llamada
desde `montecarlo.py`): escribe `inferencia_reparacion_reps*.csv.gz` con prior, post_d2 y post_d3
por celda, con θ verdadero y con cada θ̂ (descrito en `bases_datos.tex`). Prueba en `tests.py`:
las tres probabilidades promedian la tasa real y el Brier ordena prior ≥ post_d3 ≥ post_d2.

2026-11-18: humo completo en CPU sin errores (`montecarlo.py --smoke` con diseños 1–3 y
Gillingham, `por_edad_mc.py --smoke`). Para la GPU: `bash claude/niu/final/correr_gpu.sh
<fase>` (pruebas, verdad, mc, faltantes, resumen, por_edad, estado).

Pendiente: re-correr en GPU los equilibrios (`teoria.py`), el MC (`montecarlo.py`) y
`por_edad_mc.py`. `analisis/reporte_mc.py` no se copió; apunta a las salidas viejas.

## 2. Notas para corregir el texto de la tesis (no se editó `tesis.tex`)

1. **l. 150, costos de transacción.** Escribe $\mu_\tau(P_{jaw}-P_{ibx}+T_{b,\tau}+T_s)$, pero
   en las ecs. de $\nu_0$ y $B_r$ (l. 271–273) y en el código $T_b$ y $T_s$ están en utils y no
   se multiplican por $\mu$. Debería decir
   $\mu_\tau(P_{jaw}-P_{ibx})+T_{b,\tau}+T_s$.
2. **l. 313–318, nidos.** La condición "las escalas no crecen al bajar en el árbol" no se
   cumple con la calibración: $\sigma_r=0.3$ aparece dentro del nido de $w$ ($\sigma_w=0.1$), y
   en la fórmula de intercambiar el nido de compra ($\sigma_y=1$) está dentro del de $D$
   ($\sigma_D=0.3454$). El modelo está bien porque son decisiones separadas con choques propios:
   - la reparación es una **etapa posterior** con su propio choque, que entra como valor
     inclusivo $I(\sigma_r,v_1,v_0)$ en el valor de la tenencia;
   - vender o chatarrizar ($D$) entra de forma **aditiva** y es independiente de qué se compra
     (Gillingham, ec. 18).

   Conviene presentarlo así (etapas o componentes aditivos) y no como un solo árbol GEV. Las
   condiciones que sí aplican son $\sigma_w\le\sigma_y\le\sigma$ y $\sigma_D\le\sigma$; $\sigma_r$
   solo necesita ser $>0$.
3. **l. 158, edad $\bar a-1$.** Dice que en $(j,\bar a-1)$ el $w$ no es relevante. Con
   $u_w\neq0$ sí lo es: entra en la utilidad del periodo y en el precio $P_{j,\bar a-1,w}$. Lo
   único que pasa en esa edad es que ya no se repara y se llega a la terminal con certeza.
4. **l. 139, rango de $w$.** $w\in\{-\infty,\infty\}$ debe ser $w\in(-\infty,\infty)$. Lo mismo
   en la l. 158: $[-\infty,\infty]\to(-\infty,\infty)$.
5. **l. 152 vs l. 712–713, forma de $R$.** Ahora: light brown en U (caro de joven por las piezas,
   barato a media vida, sube de viejo) y heavy brown constante. La l. 152 ("U invertida o
   logarítmica, bajos de nuevos") y la fórmula de la l. 712 hay que actualizarlas.
