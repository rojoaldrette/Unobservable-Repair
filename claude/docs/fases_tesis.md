# Fases de la tesis

Pregunta central: **¿qué tanto se pueden recuperar las decisiones de reparación no
observadas** en un mercado de autos en equilibrio (Gillingham et al.) con la estrategia
de Hu & Xin (2024)?

Cada fase usa lo que produce la anterior. Las fases 3 y 4 se diseñan juntas (el estimador
define qué tiene que guardar el simulador), pero se ejecutan en este orden.

---

## Fase 1. Modelo teórico

**Qué se hace.** Escribir el modelo completo y sus propiedades.

- **Primitivas:**
  - marcas j, edades a = 1..a_max, estado s (prob. de descomponerse), estado sin coche;
  - utilidad u(j, a, s) con los parámetros de Gillingham (hogar "Low WD, Couple, Poor");
  - costos de transacción Tb y Ts (Tablas 5 y 10);
  - precio de reparación R(j, a);
  - transición s' = m(r, j, a, s) + eta (opción 2).
- **Decisiones** (`estructura_decision.md`):
  - etapa 1: keep / purge / trade, y qué coche h comprar;
  - etapa 2: reparar o no sobre h.
  - Sin chatarreo endógeno.
- **Equilibrio estacionario:** precios P(j, a, s) y distribución q tales que
  q = q M(P) y ED(P) = D(P) - S(P) = 0.
- **Resultados a mostrar o adaptar:**
  - existencia del equilibrio (Gillingham, Teorema 1, adaptado al estado continuo s);
  - el operador de Bellman es una contracción y dT/dEV = beta M (ya verificado
    numéricamente);
  - Pr(repair | h) no depende de cómo se llegó a h.
- **Argumento de identificación:** la mezcla observada
  f(s' | h) = sum_r p(r|h) F_r(s'|h) es la ec. 6.4 de Hu & Xin, con R(j, a) como
  variable excluida (Assumption 7). Discutir la frontera del grid (E[eta|s] != 0).

**Producto.** Capítulo del modelo, más `discusion_repair_o1.md` y
`estructura_decision.md` como apoyo.

**Estado.** Implementado en `claude/scripts/modelo_fin/`, incluido el equilibrio (`ED.py`).
Pendientes: ED(P) y las decisiones abiertas (precios negativos, coche terminal y
costo "no car").

---

## Fase 2. Simulación (resolver el equilibrio)

**Qué se hace.** Resolver numéricamente el modelo para una calibración base y entender
su comportamiento.

- **Solver de ED(P) = 0:** Newton sobre P, con jacobiano por `jax.jacfwd` o con la
  fórmula analítica de Gillingham (sec. 3.5). Incluye cómo tratar los precios negativos
  (piso o complementariedad).
- **Verificaciones:**
  - convergencia desde varios P iniciales (¿equilibrio único?);
  - ED = 0 a 1e-10;
  - q válida;
  - precios monótonos en a y en s.
- **Calibración propia:** R(j, a), s_repair, s_sigma, s_const, u_s, sigma_repair.
  Objetivos razonables:
  - Pr(repair) promedio cerca de 15–30%;
  - P cayendo cerca de 13% por año (dato de Gillingham);
  - tasa de accidentes parecida a la Tabla 4.
- **Estática comparativa con R (los "años").** ¿Cómo cambian P, q, las CCPs de
  reparar y las salidas al subir R? Esto ya muestra cuánta variación aporta la
  variable excluida.

**Producto.** Equilibrio base, gráficas de P(j, a, s), q y CCPs, y una tabla de
estática comparativa en R.

---

## Fase 3. Monte Carlo (generar datos)

**Qué se hace.** Usar el equilibrio como proceso generador de datos.

- **Varios "años" t = 1..T,** cada uno con su vector R_t(j, a)
  (`dataclasses.replace(g, repair_price=R_t)`). Para cada R_t se resuelve el
  equilibrio. Los años son estados estacionarios distintos, no una transición entre
  ellos (supuesto a declarar).
- **Simular un panel de N hogares por año** con M = Ω Q:
  - se guarda: estado x_t (j, a, s), elección observada (keep / purge / trade y qué h
    compró), s_{t+1}, salida del parque;
  - **se oculta r**, pero se guarda aparte como "verdad" para evaluar.
- **Diseño:** N, T, cuánto varía R entre años, R_mc réplicas (`mc_replic` = 250).

**Producto.** Función `simulate_panel(g, R_list, N, seed)` y bases simuladas.

---

## Fase 4. Estimador

**Qué se hace.** Recuperar lo no observado a partir de los datos simulados y medir qué
tan bien se recupera.

- **Etapa 1 (Hu & Xin):** identificar F_0, F_1 y p(r = 1 | h) a partir de la mezcla
  observada f(s' | h, R_t), usando la variación en R_t.
  - Opción paramétrica: máxima verosimilitud de una mezcla de normales con
    m_r = m(j, a, s) - s_repair r y pesos logit en (h, R).
  - Opción no paramétrica: el argumento de valores propios de Hu & Xin, más fiel al
    paper y más costoso.
- **Etapa 2 (estructural):** con p(r|h) y F_r recuperados, estimar los parámetros de
  utilidad y costos (mu, u, tc, ...) por verosimilitud de las elecciones observadas o
  por CCP (Hotz-Miller). La alternativa es full-solution con el equilibrio anidado
  (DNFXP de Gillingham).
- **Métricas en las R_mc réplicas:** sesgo, RMSE y cobertura de los intervalos para
  s_repair y p(r|h) por edad y s, y para los parámetros estructurales.
- **Comparaciones:**
  - estimador **ingenuo** que ignora la reparación (F única);
  - estimador **oráculo** que observa r.

  El valor de Hu & Xin está en cuánto cierra la brecha entre los dos.
- **Contrafactual de interés:** con los parámetros estimados, ¿cuánto cambian P, q y
  las salidas ante un subsidio a la reparación? Comparar con la verdad.

**Producto.** Tablas de Monte Carlo (sesgo/RMSE), gráficas de p(r|h) estimada contra
la verdadera.

---

## Fase 5. Datos simulados con zig-zag (inspecciones)

**Qué se hace.** Agregar inspecciones (`inspeccion_zigzag.md`, idea A como base y B
como extensión) y repetir las fases 2 a 4.

- **Verificar** que el modelo genera el zig-zag en las salidas sin chatarreo endógeno, y
  zig-zag en las CCPs de reparar por anticipación.
- **Estimador consciente de la inspección:**
  - con A no cambia si s se mide en la inspección;
  - con B hay que corregir los pesos en los años de inspección.
- **Estimador que ignora la inspección:** sesgo por especificación incorrecta.
- **Con B:** usar el momento de las salidas, kappa(s)(1 - pi), para sobreidentificar.

**Producto.** Gráfica de scrappage por edad (simulado) comparable a la fig. 7a de
Gillingham, y tablas de Monte Carlo con inspecciones.

---

## Fase 6. Datos daneses con costos de reparación teóricos

**Qué se hace.** Llevar el modelo a los datos reales disponibles.

- **Datos.** Gillingham publica los datos agregados por celdas (transiciones por tipo
  de hogar, coche y edad) junto con su código (sec. 5, p. 37): verificar que el
  paquete de replicación existe y qué trae. Hay probabilidad de scrap por (tipo, j, a,
  año), pero **no hay s a nivel coche ni reparaciones.**
- **Supuesto clave: en edades impares (sin inspección), scrap = accidente.** Entonces
  la tasa de scrap de la celda revela la **media** de s en esa celda:

      scrap(j, a, t) = E_t[ s | j, a ]          (a impar, o a < inspect_age_min)

  Con la opción 2, reparar a la edad a baja s_{a+1}, y por tanto el scrap a la edad
  a+1 (impar). Si R_t sube, se repara menos y el scrap en las edades impares del año
  siguiente debe subir. Esa respuesta identifica **s_repair × Δ E[p_rep]**, no cada
  factor por separado. Para separarlos se necesita la estructura del modelo (mu se
  identifica con las decisiones de trade), no solo la mezcla de Hu & Xin.
  - Ojo con la selección por supervivencia: los coches que llegan a a+1 son los que no
    se descompusieron, y esos tienen s más bajo. La media de s de los sobrevivientes no
    es la media incondicional.
  - Esta lectura se puede contrastar: comparar las tasas de scrap en edades impares con
    la logit de accidentes de la Tabla 4 de Gillingham. Gillingham sí atribuye parte
    del scrap en edades impares a chatarreo voluntario.
  - **Fuente posible de R_t real:** el índice de precios de "mantenimiento y
    reparación de vehículos" del IPC danés (COICOP 07.2.3, Statistics Denmark).
- **Implicación.** Hu & Xin no se puede aplicar literalmente, porque necesita la
  distribución de s a nivel coche, no solo su media por celda. Lo que sí se puede hacer:
  - **Integrar s** con la distribución estacionaria del modelo: por ejemplo,
    Pr(keep | j, a) = sum_s q(s | j, a) p_keep(j, a, s). Lo mismo para las salidas y
    las compras por edad.
  - **Con R(j, a) teórico** (calibrado con costos de reparación de referencia), estimar
    o calibrar el resto (dinámica de s, kappa, sigma_repair) para ajustar los momentos
    daneses: holdings por edad, keep por edad, scrappage por edad con zig-zag y
    depreciación de cerca de 13% anual.
  - **Pregunta empírica:** ¿el modelo con reparación e inspección (sin chatarreo
    endógeno) reproduce el zig-zag tan bien como el dummy de edad par de Gillingham?
- **Experimento puente (se hace antes, con el Monte Carlo).** Pregunta: ¿se recupera la
  reparación si solo se ven datos tipo daneses?
  1. Simular el panel como en la fase 3, con s coche por coche, T = 13 años y R_t
     distinto por año (idealmente con la dispersión del IPC danés de reparación).
  2. **Agregar como en los datos daneses:** borrar s y r y quedarse solo con:
     - tasas de scrap por (j, a, t), usando solo las edades sin inspección;
     - CCPs de keep / trade / purge y compras por (j, a, t);
     - holdings q por (j, a, t).
  3. Estimar el modelo estructural con s latente, integrando con q(s | j, a), por
     máxima verosimilitud de las celdas o por método de momentos. Parámetros:
     s_repair, la dinámica de s, sigma_repair y, si se puede, mu.
  4. **Comparar con la fase 4** (Hu & Xin con s coche por coche): sesgo/RMSE de
     s_repair y de p(r | j, a, s), y cuánta precisión se pierde al agregar.
  5. **Variantes:**
     - T = 5, 13, 25 años y dispersión de R_t chica o grande: ¿cuánta variación hace
       falta?
     - s_sigma fijo contra estimado: se espera que esté débilmente identificado con
       solo medias.
     - Un poco de chatarreo voluntario en edades impares (que viola el supuesto
       scrap = accidente): ¿cuánto sesgo mete?
     - Microdatos: historias de supervivencia por coche en lugar de celdas (Hu & Shum
       2012).

  Si se recupera bien, la fase 6 tiene sustento. Si no, el experimento dice qué dato
  falta (más años, más variación en R, microdatos).
- **Opcional:** usar el odómetro de las inspecciones (Tabla 6) como medida ruidosa del
  desgaste, una proxy de s observada cada dos años. Esto abre otro problema: Hu & Xin
  con un estado medido con error y observado cada dos periodos.

**Producto.** Ajuste a momentos daneses y comparación con Gillingham. Es la parte
empírica de la tesis, con la salvedad explícita de que la reparación se calibra, no se
estima.

---

## Mapa de documentos

| fase | documentos |
|---|---|
| 1 | `estructura_decision.md`, `discusion_repair_o1.md` |
| 1–4 | `modelo_fin.md` (código, calibración, equilibrio, Monte Carlo) |
| 5 | `inspeccion_zigzag.md` |
| 4, 6 | `id_modelo_vsHu.md` (qué parte es el teorema de Hu & Xin y qué parte es estructura) |
| 3 | `datasets_montecarlo.md` (diseño de las bases simuladas) |
| 4, 6 | `idea_km.md` (kilometraje como estado observado; verificación vehicular en México) |
| todas | `pruebas_robustes.md` |
