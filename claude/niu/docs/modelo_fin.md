# modelo_fin: el modelo de la tesis (documento vivo)

Este documento es la especificación vigente del modelo. Se reescribe (no se le agregan
parches al final) cada vez que se aprueba una idea, para que siempre se lea de corrido.

- **[aprobado]** = decidido con el autor.
- **[propuesta]** = valor inicial o recomendación que se puede cambiar.
- Lo que no esté aquí no es parte del modelo.

Código: `claude/niu/modelo_tesis/` (doc del código y resultados: `modelo_tesis.md`). Base:
`claude/niu/gillingham/` (`gillingham.md`).

Última actualización: 2026-10-08.

---

## 1. Qué es

Equilibrio estacionario del mercado de autos nuevos y usados de Gillingham et al. (2019)
al que se le añaden dos cosas:

- un estado de **desgaste** por coche;
- una decisión de **reparar** que el econometrista no observa.

La reparación no observada se identifica con Hu & Xin (2024): el precio de reparación es
una variable excluida que mueve la decisión de reparar pero no la transición del desgaste.

**Preguntas [aprobado]:**

1. **Sesgo de ignorar la reparación (el objetivo principal).** Si los datos vienen de este
   modelo y se estima Gillingham (sin desgaste ni reparación), ¿cuánto se sesgan sus
   parámetros (mu, utilidades, costos de transacción, accidentes)?
2. **Recuperación.** ¿Qué tanto se recuperan las decisiones de reparación no observadas, y
   qué se pierde respecto a observarlas?

**Calibración objetivo [aprobado]:** con los parámetros de Gillingham, el modelo debe dar
precios promedio por (marca, edad) que bajen de forma parecida a los de Gillingham, y una
distribución por (marca, edad) también parecida. En ambos casos se promedia sobre el
desgaste w. No se busca reproducir Gillingham exactamente apagando la reparación.

## 2. Base: Gillingham  [aprobado]

Todo lo de `niu/gillingham` se mantiene igual:

- elecciones purge / keep / trade y vender o chatarrear;
- costos de transacción y escalas;
- precios de nuevos fijos;
- tipos de hogar observados;
- parámetros de las tablas del paper.

**Diseño:**

- 2 marcas: light brown y heavy brown;
- 2 tipos de hogar con mu distinta (Low WD Couple Poor y Single Poor), sin la cual Tb y Ts
  no se separan;
- a_max = 25;
- β = 0.95.

Los cambios respecto a Gillingham son solo los de las secciones 3 y 4.

## 3. El coche: desgaste en log-odds  [aprobado]

### 3.1 Estado

Cada coche tiene una probabilidad s de descomponerse (salir del parque por accidente)
durante el año. Se trabaja con sus log-odds ℓ = log(s / (1 − s)), porque ahí el choque es
aditivo y de media cero (Hu & Xin, Assumption 2) y s siempre queda en (0, 1).

Para no gastar el grid en la tendencia por edad, que ya está en el estado a, ℓ se escribe
como la curva de Gillingham más una **desviación w**:

    ℓ = ℓ̄_j(a) + w,      ℓ̄_j(a) = acc_int_j + acc_age_j · a        (Tabla 4)

- Con w = 0 el coche tiene exactamente la probabilidad de accidente de Gillingham.
- Un coche nuevo nace con w = 0.
- El estado de un dueño es x = (j, a, w); el grid solo tiene que cubrir la dispersión de w.

### 3.2 Utilidad  [aprobado: w entra en la utilidad]

    u(j, a, w) = u0_j + u1_j a + u_w · w

- Un coche más desgastado es más incómodo o gasta más en mantenimiento menor. Así,
  reparar tiene también un beneficio inmediato, además de alargar la vida del coche.
- **u_w = −0.2 utils por unidad de w** [propuesta]. Es ~1.8 mil DKK al año para la pareja
  pobre. Es común a los tipos.
- Con u_w = 0, la única razón para reparar es sobrevivir. El autor lo probará a mano como
  ejercicio, así que el código debe permitir u_w = 0 sin cambiar nada más.

### 3.3 Transición  [aprobado: opción 2]

Durante el año, el coche h = (j, a, w) se descompone con probabilidad s(j, a, w), y la
reparación de este año no la cambia (**opción 2**). Si sobrevive:

    ℓ' = ℓ + δ_j − κ r + η,        η ~ N(0, σ_η²), independiente de r y de todo lo demás

y en la desviación (se resta la tendencia):

    w' = w + (δ_j − acc_age_j) − κ r + η

- **δ_j:** desgaste de un año sin reparar.
- **κ:** cuánto frena ese desgaste una reparación.

**Por qué la opción 2:**

- Sobrevivir depende solo del w de inicio de año, que se observa y se condiciona. Así la
  mezcla sobre r no tiene selección por supervivencia.
- w se mide al inicio de cada año (p. ej. en la inspección), así que una reparación de este
  año se ve en la medición del año siguiente.

**El desgaste es persistente:**

- No hay reversión: lo que se desgasta se queda.
- Con **δ_j ≥ κ**, un coche reparado se sigue desgastando, solo que más despacio. En
  promedio nunca mejora, y por lo tanto nunca queda mejor que nuevo.
- La forma exacta es flexible. Si hace falta, se añade una constante de degradación por
  edad o en puntos porcentuales.

**Coherencia con Gillingham:** su curva de accidentes se estimó con datos donde la gente
repara, así que debe ser el **promedio con reparaciones**:

    δ_j − κ p̄ ≈ acc_age_j,     p̄ = tasa de reparación promedio objetivo
    ⇒  δ_j = acc_age_j + κ p̄

Con δ_j ≥ κ eso exige κ ≤ acc_age_j / (1 − p̄).

**Calibración inicial [aprobado; se revisa si el modelo sale mal]:**

| parámetro | valor |
|---|---|
| σ_η | 0.15 |
| κ | 0.25 |
| p̄ | 0.3 |
| δ = acc_age + κ p̄ | (0.255, 0.281) para (LB, HB); cumple δ ≥ κ |

**κ / σ_η ≈ 1.7:** las dos campanas de la mezcla (reparó / no reparó) quedan separadas.
Si κ ≪ σ_η, la mezcla es casi unimodal y Hu & Xin identifican mal.

### 3.4 El grid de w  [aprobado: paso h = σ_η / 2]

**Discretización:** la transición se discretiza por intervalos (Tauchen). La probabilidad
de pasar al punto k es la masa de N(m, σ_η²) en [g_k − h/2, g_k + h/2], con
m = w + δ_j − acc_age_j − κr. Con esto:

- la media es exacta en el interior;
- la distribución es simétrica (tercer momento ≈ 0);
- la varianza discretizada es ≈ σ_η² + h²/12 (corrección de Sheppard).

**Paso: h = σ_η / 2 = 0.075.** Así la varianza discretizada sale inflada solo 2% y los
momentos 2.º y 3.º, con los que Hu & Xin separan la mezcla, casi no se deforman. Esto
favorece la identificación. Con h = σ_η la inflación sería de 8%, y con pasos mayores la
mezcla se deforma.

**Rango:** el que cubra el 99.5% de la masa de w en todas las edades. La dispersión de w
crece como √edad. Cuenta preliminar (simulación con prob. de reparar fija o decreciente
con la edad, pesando por sobrevivir):

| σ_η | κ | rango de w | puntos con h = σ_η/2 |
|---|---|---|---|
| **0.15** | **0.25** | **[−3.0, +2.2]** | **57-66** |
| 0.20 | 0.30 | [−3.8, +2.5] | 53-62 |
| 0.25 | 0.40 | [−4.9, +3.0] | 52-62 |

- **El número de puntos casi no depende de σ_η:** el rango y el paso escalan juntos.
- Con la calibración aprobada son **~60 puntos**.
- El rango definitivo se fija con el equilibrio resuelto. **Diagnóstico obligatorio:**
  masa en los dos puntos extremos < 0.5%. Si no se cumple, se amplía el rango con el
  mismo paso.

**Monte Carlo contra datos reales.** En el MC los datos se simulan con el mismo modelo
discreto que se estima, así que la discretización no sesga nada. Solo importa al
interpretar σ_η como la varianza de un η continuo y al llevar el modelo a datos reales.

## 4. Reparar  [aprobado]

### 4.1 Decisión

**Momento** (`claude/docs/estructura_decision.md`): después de comerciar, quien tiene un
usado h = (j, a, w) decide si lo repara:

    v_0(h) = β E[EV(x') | h, r = 0]
    v_1(h) = −μ R_t(j, a) + β E[EV(x') | h, r = 1]
    p(r = 1 | h) = logit con escala σ_rep;   la etapa 1 ve a h por W(h) = σ_rep log(e^{v_0/σ_rep} + e^{v_1/σ_rep})

- p(r | h) es la misma para quien se quedó el coche y para quien lo compró. Es la
  observabilidad parcial de Hu & Xin (sec. 6.3, ec. 6.4).
- **Quién puede reparar:** los usados de edad 1 a 23. Los nuevos no (nacen con w = 0). La
  última edad activa (24) tampoco, porque llega a la terminal de todos modos.

### 4.2 Precios de reparación

**Criterio [aprobado]:** R(j, a) y σ_rep se ajustan hasta que el patrón de reparación por
edad tenga sentido:

- poca reparación al inicio de la vida del coche (casi no hay desgaste que frenar);
- poca al final (queda poca vida que alargar);
- más en edades intermedias;
- y que nada sea absurdo (tasas, precios, edad del parque).

**Valores iniciales [propuesta, ad hoc; se cambian según se vea]:**

| | valor |
|---|---|
| R(LB, a) | 4.0 + 0.24 (a − 1) mil DKK |
| R(HB, a) | 10.0 + 0.6 (a − 1) mil DKK |
| σ_rep | 0.3 |

### 4.3 Variable excluida: regímenes  [aprobado]

R_t(j, a) = R(j, a) · e^{ζ_t}, con ζ ∈ {−0.3, 0, +0.3}. Cada valor es un **régimen**: un
"mundo" idéntico salvo por el precio de reparar (pueden leerse como años o como regiones).

- Cada régimen es su propio equilibrio estacionario: los agentes creen que su R es
  permanente.
- Es la Assumption 7 de Hu & Xin: R mueve p(r | h) pero no la transición de w.
- **Por qué 3:** Hu & Xin necesitan 2. Con la muestra fija, la identificación viene sobre
  todo de la distancia entre los extremos (±0.3). El central (la calibración base) permite
  revisar la forma de p(R) y reportar la verdad. El costo por evaluación es lineal en el
  número de regímenes.

## 5. Mercado y equilibrio  [aprobado]

**Información simétrica:** el comprador ve el desgaste. Hay un precio por celda (j, a, w),
a = 1..24, como en Gillingham pero con w.

**Nido de w:** dentro de cada (j, a), el comprador elige w con un nido de escala fija
**σ_w = 0.1** (no se estima).

- Con esa escala, los precios reflejan casi exactamente el valor del desgaste, y el efecto
  de "clonar" alternativas queda acotado a σ_w log 60 ≈ 0.4 utils.
- Robustez: σ_w = 0.05 y 0.2.

**Mercado:**

- Oferta por celda: coches no conservados y no chatarreados.
- Demanda: agregada sobre tipos con pesos f.
- Equilibrio: log D − log S = 0 por celda, con la distribución q de cada tipo
  estacionaria (por recursión en la edad, sin resolver un sistema).
- Las celdas sin oferta tienen precios sombra sin significado. Solo se reportan
  estadísticas ponderadas por masa.

**Solver:** Newton conjunto sobre z = (EV de cada tipo, P), con jacobiano denso, como en
`niu/gillingham`. Por régimen (J = 2, A = 25, 2 tipos, n_w = 60):

| | |
|---|---|
| estados por tipo | 2,883 |
| precios | 2,880 |
| incógnitas | 8,646 |
| jacobiano denso | 600 MB |

**Plan B si los tiempos no dan** [propuesta]: información asimétrica con Newton anidado
como en el paper.

- Hay un precio por (j, a) (48).
- Las creencias π(w | j, a) no son incógnitas: salen de la recursión en edad.
- El Newton de afuera es de 48×48 en lugar de ~2,900×2,900.
- Cambia la economía (selección adversa), así que sería un cambio de modelo, no solo de
  solver.

## 6. Datos e identificación  [aprobado: tres diseños]

**Siempre se observa, por hogar y año:**

- tipo y régimen;
- x = (j, a, w);
- resultado (keep / purge / trade, vendido / chatarreado);
- h comprado (con su w);
- si el coche sale del parque al año siguiente, y su w' si sobrevive.

Los precios de usados nunca se observan.

**Tres diseños**, que salen del mismo panel simulado borrando columnas, así que las
diferencias entre diseños son solo por la información:

| diseño | r (reparó) | por qué salió del parque (accidente o chatarreo voluntario) |
|---|---|---|
| **1** | se observa | se observa |
| **2** | no | se observa |
| **3** | no | no (como el registro danés de Gillingham) |

**Diseño 1, cota superior:** verosimilitud sin mezclas.

- κ, δ y σ_η salen de las transiciones de w condicionales en r.
- σ_rep sale de cómo responde la tasa de reparación a R_t.
- acc_int y acc_age salen de los accidentes observados.

**Diseño 2, la pregunta central:** cuánto se pierde al no ver r. Para cada h que
sobrevive, la transición observada es la mezcla

    f_t(w' | h) = (1 − p_t(h)) F_0(w' | h) + p_t(h) F_1(w' | h)

R_t mueve p_t(h) y no F_r, así que la variación entre regímenes identifica F_0, F_1 y
p_t(h) (Hu & Xin, ecs. 3.2-3.4 y 6.4). Se estima por máxima verosimilitud con la mezcla
adentro, y se contrasta con el primer paso no paramétrico.

**Diseño 3, como datos reales:** además de la mezcla sobre r, una salida del parque es
una mezcla de accidente y chatarreo voluntario (apéndice D de Gillingham).

- Por la opción 2, las dos mezclas no se cruzan: la probabilidad de accidente depende del
  w de inicio de año (observado), no de r. La verosimilitud es la de Gillingham "parcial"
  más la mezcla sobre r en w'.

**Estimador de Gillingham sobre los mismos datos** (pregunta 1): se agrega w (no se usa) y
se estima el modelo de `niu/gillingham` con su verosimilitud. La diferencia entre sus
estimaciones y los parámetros verdaderos es el sesgo de ignorar la reparación.

**Comparaciones:**

- Gillingham contra la verdad: el sesgo de ignorar desgaste y reparación;
- 1 contra 2: el costo de no ver r;
- 2 contra 3: el costo adicional de no distinguir el motivo de salida.

## 7. Monte Carlo  [aprobado]

**Objetivo mínimo:** recuperar el estimador y su varianza. Por parámetro y diseño: sesgo,
sd entre réplicas contra error estándar BHHH medio, y cobertura al 95%.

**Réplicas: R = 50.** Cada réplica es un panel; sobre él se estiman los tres diseños y el
modelo de Gillingham.

| R | error al estimar sd(θ̂) | error de la cobertura al 95% | sesgo detectable (2 errores estándar) |
|---|---|---|---|
| 20 | ±16% | ±4.9 pp | 0.45 sd |
| **50** | **±10%** | **±3.1 pp** | **0.28 sd** |
| 100 | ±7% | ±2.2 pp | 0.20 sd |

**Muestra: N = 10,000 hogares por régimen, K = 4 años, 3 regímenes.**

- Son ~90,000 transiciones hogar-año. Lo que cuenta es N (K − 1) × regímenes, porque el
  primer año de cada hogar solo sirve para condicionar.
- N casi no cuesta cómputo: el costo por réplica lo dominan los equilibrios. Cada réplica
  hace 4 estimaciones: los tres diseños y Gillingham.
- **Revisión con la réplica 0:** el error estándar escala como 1/√N. Si el de κ en el
  diseño 2 sale > 20% de κ, conviene subir N antes de correr las otras 49.

**Arranques:**

- Réplica 0: la verdad más 2 perturbados, para revisar máximos locales.
- Réplicas 1-49: solo la verdad.

**Avance visible:**

- Cada réplica imprime una línea al terminar: LL, convergencia y segundos por diseño.
- Cada `--print_every` réplicas (default 5) se imprime una línea de avance:
  `[avance] 10/50 réplicas, 1.2 h transcurridas, ~4.8 h restantes`.
- Dentro de cada estimación, L-BFGS imprime cada 10 iteraciones (con `-v`).
- Todo con `flush`, para que el log se vea en vivo bajo `nohup`/`tmux`.
- Los CSV se escriben al final de cada réplica, así que lo terminado no se pierde si se
  corta la corrida.

**Antes del MC:** medir el tiempo de una evaluación de la verosimilitud + gradiente con 3
regímenes. Eso decide si hace falta el plan B (sec. 5).

## 8. Decisiones abiertas

1. **Reparación al inicio de la vida.** En la corrida preliminar (`modelo_tesis.md`, sec. 4),
   Pr(reparar) baja con la edad desde 0.69 en la edad 1 (light brown). No cumple el
   criterio de la sec. 4.2. La causa: un κ constante da el mismo beneficio a un coche
   nuevo que a uno viejo, y el nuevo lo disfruta más años. Hay que elegir un mecanismo
   (opciones en `modelo_tesis.md`, sec. 4).

Valores por ajustar con el modelo resuelto:

2. u_w (sec. 3.2).
3. R(j, a) y σ_rep, con el criterio de la sec. 4.2.
4. El rango del grid de w (diagnóstico de la sec. 3.4).

## Registro de cambios

- 2026-10-08:
  - Creación. Aprobado: base Gillingham con 2 marcas, desgaste en log-odds (como
    desviación w de la curva de Gillingham) y desgaste persistente sin quedar mejor que
    nuevo.
  - Aprobado: calibración inicial (σ_η = 0.15, κ = 0.25, p̄ = 0.3), sujeta a revisión.
  - Aprobado: información simétrica. La asimétrica queda como plan B de cómputo.
  - Aprobado: paso del grid h = σ_η/2; MC con R = 50, N = 10,000 por régimen, K = 4 y
    líneas de avance periódicas.
  - Aprobado: 3 regímenes de R (ζ = −0.3, 0, +0.3).
  - Aprobado:
    - objetivo principal: el sesgo de Gillingham al ignorar la reparación;
    - calibración objetivo: precios y distribución por (marca, edad), promediados sobre w,
      parecidos a Gillingham;
    - opción 2;
    - w en la utilidad (u_w; el autor probará u_w = 0 a mano);
    - criterio para R (poca reparación al inicio y al final de la vida);
    - σ_w = 0.1;
    - quién repara (usados de edad 1 a 23);
    - tres diseños de MC según qué se observe (r; motivo de salida).
