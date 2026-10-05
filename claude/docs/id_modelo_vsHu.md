# Identificación en el modelo vs. Hu & Xin (2024)

Pregunta: ¿la tesis identifica la reparación no observada **con** Hu & Xin?

**Respuesta corta:** depende de los datos.

| datos | ¿se aplica Hu & Xin? | qué identifica |
|---|---|---|
| Monte Carlo con s por coche (fases 3–5) | **sí, su teorema** (sec. 3 + 6.2 + 6.3) | no paramétrica, punto por punto en s |
| Datos daneses por celda (fase 6) | **no su teorema; sí su Remark 5** | paramétrica, vía el problema de optimización |
| Microdatos daneses por coche | parcialmente (Apéndice B / Hu & Shum 2012) | depende de qué proxies haya por coche |

La idea central de Hu & Xin (una variable excluida mueve las CCPs y no la transición, y
eso separa la mezcla) está en los tres casos. Lo que cambia es si la identificación sale
**solo de los datos** (su teorema) o **de los datos más el modelo** (su Remark 5).

---

## 1. Qué exige el teorema de Hu & Xin

Caso binario (sec. 3.1), extendido a horizonte infinito (sec. 6.2) y a observabilidad
parcial (sec. 6.3):

- **Assumption 1:** Markov de primer orden y transición estacionaria.
- **Assumption 2:** s continuo, s' = m(r, s) + eta, con E[eta | s] = 0 y eta
  independiente de r dado s.
- **Assumption 7 (horizonte infinito):** una variable z que entra en la utilidad pero
  no en la transición. Juega el papel que "el tiempo" juega en horizonte finito.
- **Assumption 3:** dos valores z_1, z_2 en los que la media condicional de s' difiere,
  más condiciones de suavidad y orden de m(0, ·) y m(1, ·).

**El mecanismo (ecs. 3.2–3.4).** Para cada estado fijo s y cada valor de z, se calculan
de los datos los **momentos condicionales** de s_{t+1} dado s_t = s:

    media      mu_z(s)  = E[s_{t+1} | s_t = s, z]
    varianza   v_z(s)   = E[(s_{t+1} - mu_z)^2 | s_t = s, z]
    tercer m.  k_z(s)   = E[(s_{t+1} - mu_z)^3 | s_t = s, z]

- **CCP en función de las medias.** Como E[eta|s] = 0,
  mu_z = p_z m_1 + (1 - p_z) m_0, así que p_z = (mu_z - m_0) / (m_1 - m_0) (ec. 3.2).
- **m_0 y m_1 como raíces de una cuadrática.** Al diferenciar la varianza y el tercer
  momento entre z_1 y z_2 se eliminan los momentos de eta, que no dependen de z. Lo
  que queda es una cuadrática cuyas raíces son m_0 y m_1 (ec. 3.3).
- **Densidad de eta.** Con m_0, m_1 y las CCPs se despeja la densidad de eta (ec. 3.4).

Lo importante: **todo es condicional en s_t = s y usa la distribución completa de
s_{t+1}** (al menos sus primeros tres momentos). Para estimar esos objetos se necesitan
parejas (s_t, s_{t+1}) del **mismo coche**.

---

## 2. Monte Carlo: se aplica el teorema

Correspondencia entre objetos:

| Hu & Xin | este modelo |
|---|---|
| estado continuo x | s (prob. de descomponerse) |
| otros estados | (j, a) discretos: se condiciona en ellos |
| elección no observada a2 | r (reparar) |
| elección observada a1 | keep / purge / trade y qué h se compró |
| p(a2 \| x, a1) (ec. 6.4) | p(r \| h), igual para keepers y compradores (`estructura_decision.md`) |
| m(a, x) | m(r, j, d, s) = s_const_j + s_age d + s_persist s - s_repair r |
| eta | N(0, s_sigma^2), independiente de r |
| z excluida (Assumption 7) | R_t(j, a), precio de reparación por "año" |
| periodos t_1, t_2 | dos años con R distinto (con 13 años: sobreidentificado) |

Supuestos que se cumplen por construcción (opción 2):

- **Assumption 2:** reparar solo mueve la media de s' y eta no depende de r.
- **Sin selección por supervivencia en el condicional:** sobrevivir depende de s_t, que
  se condiciona, y no de r. Con la opción 1 esto falla (`discusion_repair_o1.md`).
- **Exclusión:** R_t entra solo en la utilidad de la etapa 2 (-mu R), no en F_r.

Desviaciones a declarar (están en `pruebas_robustes.md`):

1. **s discretizado en un grid.** Hu & Xin piden s continuo. El grid de 100 puntos es
   una aproximación.
2. **Frontera del grid.** Cerca de s_min y s_max las colas se acumulan, así que
   E[eta | s] != 0 y la distribución de eta depende de s. Esto viola la Assumption 2
   localmente.
3. **R_t también mueve los precios de equilibrio P_t.** No viola la exclusión, porque
   Hu & Xin solo piden que z no entre en la transición, y P no entra en F_r. Pero las
   CCPs se mueven por dos canales (directo y vía precios). Hay que decirlo, porque
   Hu & Xin están pensados para un agente individual.
4. **"Años" como equilibrios estacionarios distintos.** Los agentes creen que R_t es
   permanente. Es la lectura natural de la Assumption 7 en horizonte infinito, pero es
   un supuesto.
5. **m paramétrica (lineal).** Hu & Xin permiten m no paramétrica. Se puede usar el
   estimador no paramétrico (más fiel al paper) o explotar la forma lineal (más
   eficiente). Conviene reportar ambos.

**Conclusión:** en el Monte Carlo la tesis **sí aplica Hu & Xin** en sentido estricto, en
un modelo de equilibrio con observabilidad parcial. Esa es la contribución central.

---

## 3. Datos daneses por celda: no se aplica el teorema

### Qué hay

Por (tipo de hogar, j, a, año): CCPs de keep / trade / purge, compras, holdings y tasa de
scrap. Supuesto: en edades sin inspección, scrap = accidente, así que

    scrap(j, a, t) = E_t[ s | j, a ]

### Qué falta

El teorema necesita mu_z(s), v_z(s) y k_z(s): momentos de s_{t+1} **dado s_t = s**. Las
celdas dan una sola cantidad por celda y año, la media de s **integrada** sobre todos
los coches de la celda:

    E_t[s_{t+1} | j, a+1]  =  ∫ q_t(s | j, a) * (1 - s) * E[s' | s, r] ... ds   (con selección)

- **No se condiciona en s_t.** Las celdas mezclan coches con s distintos.
- **No hay segundo ni tercer momento.** El scrap es binario por coche, así que solo
  revela la media de s.

Ejemplo (ver `fases_tesis.md`): "la mitad repara con efecto 0.06" y "todos reparan con
efecto 0.03" dan la misma media de s'. Solo los distingue la varianza (hay
s_repair^2 p(1-p) de varianza extra en el primer caso), y las celdas no la ven.

### ¿Y si cada año es una muestra con distinto R? (pseudo-panel por cohorte)

La idea: los coches (j, a) del año t son, salvo los que salen, los mismos coches
(j, a+1) del año t+1. Cada cohorte (j, año de fabricación) es una unidad seguida en el
tiempo (un pseudo-panel, como en Deaton 1985), y los 13 años dan 13 valores de la
variable excluida R_t.

**Lo que esto sí da (y está bien):** la variación en z que pide la Assumption 7. Tratar
los años como valores distintos de z es exactamente lo que hace Hu & Xin en horizonte
infinito. Con 13 valores sobra: el caso binario necesita 2.

**Lo que no da:** la unidad cuyo s se sigue es la **cohorte**, no el coche. Dentro de la
cohorte una fracción p_t repara y el resto no. Con muchos coches, eta se promedia y la
media de la cohorte se mueve de forma casi determinista:

    s̄_{t+1} ≈ m_0(s̄_t) - s_repair * p̄_t(R_t)        (+ selección por supervivencia)

- **Coche por coche,** s_{t+1} es una **mezcla** de dos campanas (reparó / no reparó).
  Hu & Xin separan m_0, m_1 y p con el 2.º y 3.er momento de esa mezcla.
- **Para la cohorte,** s̄_{t+1} es un **promedio**: la mezcla colapsa a su media. La
  varianza entre años de s̄ es ruido muestral más el efecto de R, no la varianza
  s_repair^2 p(1-p) de la mezcla.

Queda solo la ec. 3.2 en medias, con una incógnita p_t por año:

    mu_t = m_0 - s_repair * p_t,    t = 1..13   ->  13 ecuaciones, 13 + 2 incógnitas

Ejemplo:

| | año A (R bajo) | año B (R alto) |
|---|---|---|
| hipótesis 1: efecto 0.06 | p = 0.50 -> media 0.190 | p = 0.25 -> media 0.205 |
| hipótesis 2: efecto 0.03 | p = 1.00 -> media 0.190 | p = 0.50 -> media 0.205 |

Las dos dan las mismas medias. Coche por coche se distinguirían: en la hipótesis 1, en
el año A hay dos grupos separados por 0.06; en la 2, un solo grupo.

**Cómo se identifica entonces: con la forma de p en R.** Si p_t = Lambda(alpha - beta
R_t) (logit reducido, sin resolver la Bellman), quedan 4 parámetros (m_0, s_repair,
alpha, beta) y 13 ecuaciones. La hipótesis 2 exige que p se sature en 1 con R bajo, y
eso dibuja una curva media-vs-R distinta a la de la hipótesis 1. La identificación
viene de la **curvatura** del logit:

- Si R_t varía poco entre años, el logit es casi lineal en ese rango y solo se identifica
  el producto s_repair * beta.
- Hace falta variación de R suficientemente amplia para "ver" la curvatura. El
  experimento puente lo mide: RMSE contra dispersión de R_t.

Esta vía es más débil que el teorema (supone la forma funcional de p) pero más fuerte que
el Remark 5 completo: no necesita la Bellman, solo que p sea monótono y logit en R.

**¿Sirve la fluctuación de la media de la cohorte entre años como "segundo momento"?**
No. Hu & Xin usan la varianza **entre coches** con el mismo s en un mismo año. La
fluctuación **entre años** de la media de la cohorte es otra cosa:

1. **Si R_t es constante,** la media solo fluctúa por ruido muestral (y choques
   agregados). No hay variable excluida que mueva p, así que no hay nada que separar.
2. **El ruido muestral casi no contiene la mezcla.** Lo que se observa es una tasa de
   scrap, es decir, la media de Bernoulli(s_i). Su varianza es

       Var(tasa) = [ s̄(1 - s̄) - Var_i(s_i) ] / N

   La dispersión de s entre coches (donde vive la mezcla) entra dividida entre N y con
   signo negativo. Con N = 5,000, s̄ = 0.05 y una mezcla s_repair = 0.06, p = 0.5:
   - varianza binomial 9.5e-6;
   - contribución de la mezcla 1.8e-7, es decir, **1.9%** del total.

   Con 13 años, una varianza se estima con un error relativo de cerca de 41%. La señal
   es indetectable.
3. **Si R_t varía,** la información está en cómo se mueve la **media** con R_t (el
   primer momento). Eso lleva a la identificación por curvatura del logit descrita
   arriba, no al teorema.

### Qué sí se puede: Remark 5 de Hu & Xin

Los mismos autores lo prevén (Apéndice B, Remark 5): cuando los supuestos fuertes no se
cumplen,

> "we can connect the unobserved choice probabilities and the latent state transition
> rules through (1) the observed state transition process, and (2) the agent's
> optimization problem. This provides us with a system of nonlinear equations, through
> which we can locally identify the CCPs and the state transition rules conditional on
> the choice. Imposing parametric assumptions on the utility function and/or the state
> transition rules may further reduce the dimension of the parameter space."

Eso es exactamente lo que haría la fase 6:

1. **Las CCPs de reparar no son libres:** salen de la Bellman, p(r | h; R_t, mu,
   sigma_repair, s_repair, ...).
2. **La variación de R_t** (13 años) mueve esas CCPs y, a través de ellas, el scrap
   del año siguiente en las edades impares, las decisiones de keep / trade (vía el valor
   de poder reparar) y los holdings.
3. **Se estima el modelo con s latente**, integrando con la distribución estacionaria
   q_t(s | j, a), igual que Gillingham estima tasas de accidente sin observar accidentes.

Diferencias con el teorema:

| | teorema (Monte Carlo) | Remark 5 (celdas danesas) |
|---|---|---|
| tipo de identificación | no paramétrica, global, punto por punto en s | paramétrica, local |
| depende del modelo de decisión | no (las CCPs se identifican antes de la Bellman) | sí (las CCPs las da la Bellman) |
| robustez a errores en la utilidad | alta | baja: si la utilidad está mal, los pesos de la mezcla también |
| s_sigma | identificado (densidad de eta) | débil: las medias casi no lo informan |
| prueba de "¿hay reparación?" (sec. 6.3) | sí, con los datos | solo dentro del modelo |

La pérdida más importante es la segunda fila. Con el teorema, la reparación se
identifica **sin** suponer cómo deciden los hogares, y luego se puede contrastar la
Bellman. Con el Remark 5, la identificación descansa en la Bellman.

### Cómo cuantificarlo

El experimento puente (`fases_tesis.md`, fase 6) simula con s por coche, agrega a celdas
y estima con el Remark 5. La distancia entre su RMSE y el de la fase 4 mide lo que se
pierde al pasar del teorema al Remark 5.

---

## 4. Microdatos por coche: punto intermedio

Si se consigue el registro coche por coche (Statistics Denmark), sigue sin haber s, pero
hay historias por coche:

- **Supervivencia en años consecutivos:** Pr(sobrevivir a, a+1, ...) depende de la
  persistencia de s, no solo de su media. s es un estado latente markoviano: es el
  terreno de Hu & Shum (2012), identificación con estados no observados.
- **Apéndice B de Hu & Xin:** con **dos** variables de estado discretas cuyas
  transiciones son independientes dado r (Assumption 9), la reparación se identifica con
  una descomposición de valores propios usando solo dos periodos. Candidatos a revisar:
  sobrevivir y algún resultado de la inspección o el kilometraje del odómetro. La
  independencia condicional en r es fuerte (el uso del coche afecta las dos), así que
  esto es por explorar, no algo ya resuelto.

---

## 5. Cómo contarlo en la tesis

1. **Contribución metodológica (Monte Carlo):** se aplica el teorema de Hu & Xin con
   observabilidad parcial dentro de un modelo de equilibrio de mercado de autos, con el
   precio de reparación como variable excluida. Se mide qué tanto se recupera la
   reparación no observada y cómo afecta eso a los contrafactuales.
2. **Puente:** el mismo Monte Carlo agregado a datos tipo daneses muestra cuánto se
   pierde al pasar de la identificación no paramétrica a la del Remark 5.
3. **Aplicación (fase 6):** con datos daneses y costos de reparación teóricos, el modelo
   se estima vía el Remark 5. Hay que decir explícitamente que ahí la identificación
   descansa en la estructura y no solo en los datos.

Esto no debilita la tesis: deja claro qué parte es "Hu & Xin puro" y qué parte es
"Hu & Xin + estructura", y el experimento puente cuantifica la diferencia.
