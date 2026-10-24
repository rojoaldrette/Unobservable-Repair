# Estimar con decisiones no observadas: un paso (lo que hago) vs. dos pasos (Hu & Xin, Remark 2)

Este documento solo trata una pregunta: **cómo se obtiene Pr(reparar | h) si la reparación nunca
se ve**, y en qué se diferencia el estimador que uso (máxima verosimilitud de información
completa, en un paso) del estimador de dos pasos que proponen Hu & Xin (2024) en su Remark 2.
También explica qué garantiza la identificación en cada caso.

Se supone que solo se conoce lo básico de máxima verosimilitud (MV). Todo lo demás se explica.

Referencias: Hu & Xin (2024), `docs/papers/unobs_choice.pdf`, secs. 3.1, 4, 6.2 y 6.3. Código:
`modelo_tesis/bellman.py` (`repair_stage`, `w_transition`) y `modelo_tesis/ll_estim.py`
(`cell_logp`). Notación y diseños: `docs/loglikelihood.md`.

---

## 0. Resumen en una tabla

| | **Un paso (lo que hago)** | **Dos pasos (Hu & Xin, Remark 2)** |
|---|---|---|
| Qué se maximiza | Una sola verosimilitud con *todos* los parámetros θ | Paso 1: verosimilitud de la transición de w. Paso 2: un criterio para las preferencias |
| De dónde sale p(h) = Pr(reparar \| h) | Del modelo: logit de la etapa 2 con θ, después de resolver Bellman y equilibrio | Del paso 1: se recupera de los datos (la mezcla), **sin** usar el modelo de decisión |
| Forma de la transición de w | Paramétrica: w' = w + δ − acc_age − κr + η, η ~ Normal | No paramétrica: w' = g(r, w) + η, con g y la distribución de η desconocidas (sieve) |
| Qué identifica la mezcla | La variable excluida R_t **y además** la forma funcional (normalidad, logit) | Solo la variable excluida R_t (más supuestos débiles) |
| Hay que resolver el modelo | Sí, en cada evaluación (Bellman + equilibrio, 3 regímenes) | Paso 1 no. Paso 2 depende de cómo se haga (sec. 5.3) |
| Si la normalidad está mal | Sesgo en todo θ | El paso 1 no se ve afectado |
| Eficiencia (si el modelo está bien) | Máxima: es MV de información completa | Menor; errores estándar del paso 2 deben corregirse por el paso 1 |
| ¿Está implementado? | Sí (`ll_estim.py`) | No |

La idea central: **Hu & Xin dicen qué se puede aprender de los datos (identificación); el
estimador de un paso y el de dos pasos son dos maneras de usar esa información.**

---

## 1. Lo mínimo de máxima verosimilitud que hace falta

### 1.1 La verosimilitud

Dado un modelo con parámetros θ, la verosimilitud de una observación es la probabilidad que el
modelo le asigna a lo que se observó. Con observaciones independientes, la log-verosimilitud es
la suma de los logaritmos:

    ℓ(θ) = Σ_i log Pr(dato_i | θ)

y el estimador de MV es el θ que la maximiza.

### 1.2 Qué pasa cuando una variable no se observa

Supón que el modelo dice cómo se generan (r, w') juntos, pero solo se ve w'. Lo que se ve tiene
probabilidad

    Pr(w' | θ) = Σ_r Pr(r | θ) · Pr(w' | r, θ)

(ley de probabilidad total: se suman todas las maneras de llegar a w'). La verosimilitud usa esta
suma, **dentro** del logaritmo:

    ℓ(θ) = Σ_i log [ Σ_r Pr(r | θ) · Pr(w'_i | r, θ) ]

Esta fórmula es correcta, pero no dice si los datos alcanzan para encontrar θ. Eso es la
identificación.

### 1.3 Identificación

Un parámetro está **identificado** si dos valores distintos del parámetro generan distribuciones
distintas de lo observable. Si θ_A y θ_B producen exactamente la misma distribución de los datos,
ni con infinitos datos se pueden distinguir: la verosimilitud poblacional tiene el mismo valor en
los dos y no hay un máximo único.

Con decisiones no observadas el riesgo es claro: "repara poco, y reparar ayuda mucho" y "repara
mucho, y reparar ayuda poco" pueden producir casi el mismo w' promedio. Hu & Xin dan condiciones
para que eso no pase.

---

## 2. El objeto observable: una mezcla

Fija una tenencia h = used(j, d, w) (marca, edad y desgaste del coche con el que se maneja el
año), un régimen t (que fija el precio de reparación R_t = R · e^{ζ_t}) y un tipo de hogar τ.
Condicional en que el coche **sobrevive** el año, el w' del año siguiente se distribuye así:

    f_t(w' | h) = (1 − p_t(h)) · F_0(w' | j, w)  +  p_t(h) · F_1(w' | j, w)

- F_0: distribución de w' si **no** se repara.
- F_1: distribución de w' si **sí** se repara.
- p_t(h): probabilidad de reparar (el CCP que no vemos).

Es una **mezcla de dos distribuciones**, con pesos p_t(h) y 1 − p_t(h). Se ve f_t (porque se ven
h y w'), pero no los tres ingredientes por separado. Es la ec. (3.1) de Hu & Xin, y en su versión
de observabilidad parcial, la ec. (6.4).

En mi modelo:

    w' = g(r, w) + η,   g(r, w) = w + δ_j − acc_age_j − κ · r,   η ~ N(0, σ_η²)

(calibración: κ = 0.25, σ_η = 0.15, δ_j − acc_age_j = κ · 0.3 = 0.075). Reparar baja la media
de w' en κ y no cambia nada más. `w_transition` discretiza esta normal en el grid de w.

**¿Por qué condicionar en sobrevivir no estropea nada?** Porque la probabilidad de descomponerse
s(h) no depende de r (la "opción 2" de `modelo_fin.md`). Entonces sobrevivir no dice nada sobre r,
y los pesos de la mezcla entre los que sobreviven siguen siendo p_t(h). Con la "opción 1"
(reparar baja s ya este año) los pesos entre sobrevivientes serían
p_t(1 − s_1) / [p_t(1 − s_1) + (1 − p_t)(1 − s_0)], que sigue siendo una mezcla pero con otros
pesos. La identificación se adapta, pero las fórmulas cambian.

**El papel de la decisión observada.** Hu & Xin (sec. 6.3) separan la decisión en a¹ (observada:
keep / purge / trade y qué se compra) y a² (no observada: reparar). La ec. (6.4) condiciona la
mezcla en a¹. En mi modelo la etapa 2 depende de la etapa 1 solo a través de la tenencia h que
resulta, así que condicionar en (x, a¹) es lo mismo que condicionar en h.

---

## 3. Identificación: qué la garantiza (Hu & Xin, sec. 3.1 y sec. 6.2)

Esta sección es la base de los dos estimadores. Es un argumento **no paramétrico**: no supone
que η sea normal, ni que g sea lineal, ni que p venga de un logit. La pregunta es: si se
conociera f_t(w' | h) exactamente (infinitos datos), ¿se podrían recuperar F_0, F_1 y p_t(h)?

### 3.1 Con un solo régimen no se puede

Sin suponer normalidad, las incógnitas en un h fijo son g(0, w), g(1, w), p, y toda la
distribución de η. Usando solo los tres primeros momentos de w' (media, varianza, tercer momento
centrado), con Δ = g(1, w) − g(0, w):

    media:          μ = g(0, w) + p · Δ
    varianza:       v = Var(η) + p (1 − p) Δ²
    tercer momento: k = E[η³] + p (1 − p)(1 − 2p) Δ³

Son 3 ecuaciones y 5 incógnitas (g0, g1, p, Var(η), E[η³]). Los momentos más altos añaden una
incógnita más cada uno (E[η⁴], ...). **Siempre faltan ecuaciones:** hay infinitas combinaciones
de (componentes, pesos) que dan exactamente la misma f.

### 3.2 La variable excluida (Assumption 7)

En horizonte finito, Hu & Xin usan el tiempo: los CCPs cambian con la edad del agente pero la
transición no. En horizonte infinito (sec. 6.2, que es mi caso) hace falta una variable que:

1. **mueva los CCPs**: R_t cambia lo que cuesta reparar, así que cambia p_t(h);
2. **no entre en la transición**: F_0 y F_1 no dependen de R_t. El precio del taller no cambia
   cuánto mejora el coche al repararlo, ni el choque η.

Con dos regímenes t = 1, 2 se observan dos mezclas **con las mismas componentes y distintos
pesos**:

    f_1(w' | h) = (1 − p_1) F_0 + p_1 F_1
    f_2(w' | h) = (1 − p_2) F_0 + p_2 F_1

### 3.3 Cómo se recuperan los pesos (ecs. 3.2-3.3)

Con dos regímenes hay 6 momentos (3 por régimen) y 6 incógnitas: g0, g1, p_1, p_2, Var(η),
E[η³]. Var(η) y E[η³] son las mismas en los dos regímenes, porque η no depende de r ni de R_t.
Por eso, al restar la varianza (o el tercer momento) de un régimen menos la del otro, η
desaparece:

    v_1 − v_2 = Δ² [ p_1(1 − p_1) − p_2(1 − p_2) ]
    k_1 − k_2 = Δ³ [ p_1(1 − p_1)(1 − 2p_1) − p_2(1 − p_2)(1 − 2p_2) ]

Junto con las dos medias quedan 4 ecuaciones en (g0, g1, p_1, p_2). Hu & Xin muestran que g0 y
g1 son las dos raíces de una ecuación cuadrática cuyos coeficientes son funciones de los momentos
observables (ec. 3.3). Después, cada peso sale de la media (ec. 3.2):

    p_t(h) = (μ_t − g0) / (g1 − g0)

**Ejemplo numérico con mi calibración** (w = 0 y, para el ejemplo, p_1 = 0.2, p_2 = 0.4):
g0 = 0.075, g1 = −0.175, Var(η) = 0.0225, E[η³] = 0. Los momentos observables son

| régimen | media μ_t | varianza v_t | tercer momento k_t |
|---|---|---|---|
| 1 | 0.025 | 0.0325 | −0.0015 |
| 2 | −0.025 | 0.0375 | −0.00075 |

Al resolver numéricamente las 6 ecuaciones en las 6 incógnitas (lo comprobé con 3,000 puntos de
arranque) salen **exactamente dos soluciones**:

- (g0, g1, p_1, p_2) = (0.075, −0.175, 0.2, 0.4), que es la verdadera;
- (−0.175, 0.075, 0.8, 0.6), que es la misma con las etiquetas cambiadas: "no reparar" se llama
  "reparar" y viceversa.

### 3.4 Quién es quién (Assumption 3(iii))

Los datos no saben cuál componente es "reparar". Hu & Xin piden una condición de orden en algún
punto. En mi modelo la economía la da: **reparar baja el desgaste**, g(1, w) < g(0, w) (κ > 0).
Con eso se descarta la segunda solución.

### 3.5 Cómo se recuperan las componentes completas (ec. 3.4)

Ya conocidos p_1 ≠ p_2, las dos mezclas forman un sistema lineal de 2 × 2 en F_0 y F_1, punto por
punto en w':

    F_1(w') = [ f_1(w') (1 − p_2) − f_2(w') (1 − p_1) ] / (p_1 − p_2)
    F_0(w') = [ f_2(w') p_1 − f_1(w') p_2 ] / (p_1 − p_2)

Se divide entre p_1 − p_2: **si el precio de reparación no mueve la probabilidad de reparar, no
hay identificación.** Esa es la condición de relevancia de la variable excluida.

### 3.6 Lista de lo que garantiza la identificación, y si se cumple en mi modelo

| Supuesto (Hu & Xin) | Qué pide | En mi modelo |
|---|---|---|
| A1(i) Markov | Mañana depende solo del estado y la decisión de hoy | Sí, por construcción |
| A1(ii) Estacionariedad | F_r no cambia con el tiempo | Sí (horizonte infinito, estacionario) |
| A2 Transición aditiva | w' = g(r, w) + η, con E[η \| w] = 0 y η independiente de r dado w | Sí en el interior del grid. **En las fronteras no**: el recorte del grid hace E[η \| w] ≠ 0. El piso eps_F también lo altera un poco (1e-10, despreciable) |
| A7 Variable excluida | R_t mueve p_t, no mueve F_r | Sí: R_t solo entra en la utilidad de reparar (`repair_stage`), no en `w_transition` |
| Relevancia | p_1(h) ≠ p_2(h) | Sí mientras σ_rep sea finito y la reparación no sea segura (p ≈ 1) ni imposible (p ≈ 0) en ese h |
| A3(i) | Las medias difieren entre regímenes salvo en un conjunto de medida cero | Equivale a κ ≠ 0 más la relevancia |
| A3(iii) Orden | Saber cuál componente es cuál | κ > 0: reparar baja w |
| Opción 2 | s no depende de r | Sí; si no, cambian los pesos (sec. 2) |

**Lo que el argumento entrega.** Con estos supuestos, a partir de las mezclas observadas se
recuperan, para cada h, régimen y tipo: p_t,τ(h), g(0, w), g(1, w) y la distribución de η. Es
decir, el CCP de reparar **sí se identifica de los datos, sin usar el modelo de decisión.** Las
preferencias (μ, u, costos de transacción, σ_rep) se identifican después a partir de esos CCPs,
como en un modelo dinámico con decisiones observadas (Magnac & Thesmar, 2002; Hu & Xin sec. 3.4).

---

## 4. Lo que hago: MV de información completa en un paso

### 4.1 La verosimilitud

Para el diseño 2 (r no observado; `cell_logp`, rama `design != 1`), una transición
h_{k−1} → (x_k, o_k, h_k) aporta

    Pr(x, o, h_k | h; θ) = [ (1 − p_t(h; θ)) · Pr(x | h, r=0; θ) + p_t(h; θ) · Pr(x | h, r=1; θ) ]
                           · C_t[x, o; θ] · buy_t(h_k; θ)^{1(trade)}

donde:

- Pr(x | h, r; θ) = s(h; θ) si el coche murió, o (1 − s(h; θ)) · F_r(w' | j, w; θ) si sobrevivió,
  con F_r **normal** de media w + δ − acc_age − κr (paramétrica);
- p_t(h; θ) = Pr(reparar | h) es el **logit de la etapa 2**, calculado con el modelo:

      p_t(h; θ) = Λ( [ −μ R_t(j, d) + cv_1(h) − cv_0(h) ] / σ_rep )

  donde Λ es la logística y cv_r(h) son los valores de continuación si se repara o no
  (`repair_stage`, `bellman.py:114-122`). Para tener cv_r hay que resolver la ecuación de Bellman
  **y** el equilibrio de precios P_t(θ) del régimen t;
- C_t y buy_t son los CCPs de comercio (keep / purge / trade, vender / chatarrear, qué comprar),
  también del modelo resuelto.

Esa es la ec. (4.1) de Hu & Xin con a = (a¹, a²), a¹ observado: la suma sobre la decisión no
observada va dentro del log.

### 4.2 Qué significa "en un paso"

Hay **un solo** vector θ = (μ, u0, u1, costos de transacción, σ_sell, acc_int, acc_age, u_w, δ,
κ, σ_η, σ_rep) y una sola maximización. Cada vez que el optimizador prueba un θ:

1. resuelve el equilibrio (P_t, EV_t) de los 3 regímenes;
2. calcula p_t(h; θ), C_t, buy_t y F_r(θ);
3. evalúa ℓ(θ) sumando sobre r dentro del log.

Nunca se calcula un "p̂(h)" a partir de los datos. **El CCP de reparar es siempre el que implica
el modelo para ese θ.** En el sentido del modelo sí hay CCP de reparar; lo que no hay es un CCP
estimado por frecuencias o por una mezcla no paramétrica.

### 4.3 Dónde entra Hu & Xin en este estimador

Solo en el argumento de identificación (sec. 3): garantiza que la información de los datos
basta para separar "cuánto se repara" de "cuánto sirve reparar". El estimador no usa sus
fórmulas (3.2)-(3.4). Es el caso paramétrico de su estimador (4.1): ellos aproximan g y la
densidad de η con un *sieve* (polinomios cuyo grado crece con la muestra); yo fijo g lineal y η
normal desde el inicio. En su Tabla 1 usan polinomios de grado 3 para g y de grado 5 para la raíz
de la densidad de η.

### 4.4 Un matiz de identificación: aquí hay más de lo necesario

En el estimador de un paso, p_t(h) está identificado **por dos vías a la vez**:

1. **Por la variable excluida** (sec. 3): la variación de R_t entre regímenes, como en Hu & Xin.
2. **Por la forma funcional**:
   - una mezcla de dos normales con la misma varianza ya está identificada con una sola
     distribución (resultado clásico de mezclas finitas de normales). Con normalidad, incluso
     un solo régimen bastaría en principio;
   - p_t(h; θ) no es libre en cada h: es un logit con pocos parámetros (μ, σ_rep) y cv_r(h) sale
     del modelo. κ y σ_η son comunes a todos los h. Esto liga miles de celdas con unos pocos
     números.

Consecuencias:

- **A favor:** más precisión. Toda la estructura ayuda a separar la mezcla.
- **En contra:** si η no es normal o g no es lineal en r, la vía 2 empuja hacia valores
  equivocados y el sesgo pasa a todo θ. La vía 1 (Hu & Xin) es la que hace que la identificación
  **no dependa** de la normalidad. Esto es lo que se puede defender en la tesis: el modelo está
  identificado aunque se relaje la forma funcional.
- **Identificación débil en la práctica:** con mi calibración las dos componentes están cerca
  (κ / σ_η = 0.25 / 0.15 ≈ 1.7 desviaciones estándar). Con un solo régimen y solo la normalidad,
  separarlas costaría mucha muestra. La variación de R_t es lo que da la fuerza.

---

## 5. El estimador de dos pasos (Hu & Xin, Remark 2)

### 5.1 Paso 1: transición y CCPs de reparar, sin modelo de decisión

Se maximiza solo la parte de la verosimilitud que corresponde a la **transición de w** (ec. 4.5):

    ℓ_1(g, f_η) = Σ_obs log [ Σ_r f_η( w'_i − g(r, w_i) ) · p_t(r | h_i; g, f_η) ]

- g(r, w) y la densidad de η, f_η, se aproximan con sieves (polinomios).
- Los CCPs **no son parámetros libres ni salen de un logit**: se calculan con la fórmula cerrada
  (3.2), p_t(h) = (μ_t(h) − g(0, w)) / (g(1, w) − g(0, w)), donde μ_t(h) es la media condicional
  observada de w' (estimada directamente de los datos).
- No hay utilidades, no hay Bellman, no hay equilibrio.

Resultado del paso 1: p̂_t,τ(h), ĝ(0, ·), ĝ(1, ·), f̂_η. Es decir, F̂_0, F̂_1 y los CCPs de reparar
**estimados de los datos.**

Los CCPs de comercio (keep / purge / trade, qué se compra) no necesitan nada de esto: se observan,
y se estiman por frecuencias, Pr̂(o | x) y Pr̂(h | trade).

### 5.2 Paso 2: preferencias a partir de los CCPs (Hotz-Miller y relacionados)

Con todos los CCPs y las transiciones ya estimados, se recuperan las utilidades con los métodos
de CCP (Hotz & Miller, 1993; Hotz et al., 1994; Aguirregabiria & Mira, 2002). La idea básica es
**invertir los logits**. Para la reparación:

    σ_rep · log[ p̂_t(h) / (1 − p̂_t(h)) ]  =  −μ R_t(j, d) + cv_1(h) − cv_0(h)

El lado izquierdo es un dato (sale del paso 1) y el derecho es función de θ. Los valores de
continuación cv_r se pueden escribir en términos de los CCPs y las transiciones estimadas sin
resolver la ecuación de Bellman (esa es la gracia de Hotz-Miller). Con la variación de R_t entre
regímenes en el mismo h se ve directamente cuánto responde la reparación al precio, lo que
informa sobre μ / σ_rep.

### 5.3 La complicación propia de mi modelo: los precios

Hu & Xin trabajan con un agente único. En mi modelo, C, buy y cv dependen de los precios de
usados P_t(j, a, w), que **no se observan**. En el paso 2 habría que:

- **recuperar P de los CCPs**, invirtiendo buy: la probabilidad de comprar used(j, d, w) es un
  logit anidado en u − μP + W, y se puede despejar P dado θ; o
- **resolver el equilibrio** dentro del paso 2, con lo que se pierde buena parte de la ventaja
  computacional.

Este punto no está resuelto ni implementado; habría que diseñarlo.

### 5.4 Errores estándar

En el paso 2, p̂ y F̂ se tratan como si fueran datos, pero son estimaciones con error. Los errores
estándar del paso 2 deben corregirse por la incertidumbre del paso 1 (corrección tipo Murphy-Topel,
o bootstrap de los dos pasos juntos). En un paso esto no hace falta: la matriz BHHH de la
verosimilitud completa ya incluye todo.

---

## 6. Las diferencias, una por una

### 6.1 De dónde sale p(h)

- **Un paso:** p(h; θ) = logit del modelo. Siempre es coherente con las preferencias, el
  equilibrio y la transición, porque los tres salen del mismo θ.
- **Dos pasos:** p̂(h) sale de la mezcla, celda por celda o suavizado. No está obligado a ser
  coherente con ningún logit: es lo que dicen los datos.

### 6.2 Supuestos que se necesitan

- **Un paso:** todos los de la sec. 3 **más** normalidad de η, g lineal en r, logit en la etapa 2
  y el modelo completo de equilibrio.
- **Dos pasos, paso 1:** solo los de la sec. 3 (A1, A2, A7, relevancia, orden). Sin normalidad y
  sin modelo de decisión.

### 6.3 Qué pasa si algo está mal especificado

- **Un paso:** el error se propaga a todo θ. Por ejemplo, si η tuviera colas pesadas, el
  estimador podría confundir colas con reparación y sesgar κ, σ_rep y, a través del equilibrio,
  las preferencias.
- **Dos pasos:** el paso 1 es robusto a errores en las preferencias y en la forma de η. Un error en
  el paso 2 no contamina p̂.

### 6.4 Precisión

- **Un paso:** si el modelo es correcto, es el estimador más eficiente posible (MV de información
  completa). Usa la estructura para compartir información entre celdas.
- **Dos pasos:** menos eficiente. Además, en muestras finitas los CCPs estimados con ruido sesgan
  el paso 2 (un problema conocido de Hotz-Miller; Aguirregabiria & Mira lo mitigan iterando).

### 6.5 Datos que se necesitan, en concreto para este MC

El paso 1 no paramétrico necesita estimar una mezcla por cada (t, τ, h). Con h reparable hay
2 marcas × 23 edades × 71 valores de w = 3,266 celdas, por 2 tipos y 3 regímenes. Con 90,000
transiciones por réplica, quedan **a lo más unas 4-5 observaciones por celda en promedio** (y
muchas vacías, porque la distribución estacionaria se concentra). Celda por celda no se puede: hay
que suavizar en (d, w) con un sieve, que es justo lo que proponen Hu & Xin. El estimador de un
paso no tiene este problema porque p(h; θ) depende de pocos parámetros.

### 6.6 Costo computacional

- **Un paso:** caro. Cada evaluación de ℓ resuelve 3 equilibrios y necesita sus derivadas
  implícitas (`LLEval`, `dz_dx`). Por eso corre en GPU.
- **Dos pasos:** el paso 1 es barato (no hay modelo económico). El costo del paso 2 depende de
  cómo se resuelva lo de los precios (sec. 5.3).

---

## 7. Qué ganaría la tesis con el paso 1, aunque se estime en un paso

Aunque la estimación principal siga en un paso, el paso 1 de Hu & Xin sirve como **contraste**:

1. **Comparar CCPs.** Graficar p̂_t(h) (no paramétrico, del paso 1) contra p_t(h; θ̂) (del modelo
   estimado), por edad y desgaste. Si se parecen, el logit y la normalidad no están forzando el
   resultado.
2. **Prueba de relevancia de la decisión no observada** (Hu & Xin, final de la sec. 6.3): si la
   distribución de w' dado h no cambia entre regímenes, la reparación (o su respuesta al precio) no
   importa empíricamente. Se puede probar **antes** de estimar el modelo estructural.
3. **Robustez de κ.** Comparar ĝ(0, w) − ĝ(1, w) del paso 1 con κ̂ del paso único, y ver si el
   efecto de reparar es constante en w y en la edad, como supone el modelo.

En el MC actual, con un DGP paramétrico, ambos estimadores son consistentes; el de un paso debería
ser más preciso. La diferencia aparecería con datos reales o con un DGP mal especificado (por
ejemplo, generar η con colas pesadas y estimar con normalidad).

---

## 8. En una frase

**Hu & Xin garantizan que la probabilidad de reparar está escondida en los datos (en cómo cambia
la mezcla de transiciones de w cuando cambia el precio de reparar). El estimador de dos pasos la
saca de ahí primero y estima las preferencias después. El mío no la saca nunca por separado:
busca el θ cuyo logit de reparación, junto con todo lo demás, hace más probables los datos. La
identificación de Hu & Xin es lo que asegura que ese θ es único sin depender de la normalidad.**
