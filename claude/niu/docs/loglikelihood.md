# La verosimilitud de cada diseño

Cómo se construye la log-verosimilitud con la que se estima el modelo de la tesis en los tres
diseños del MC, qué parte viene de Hu & Xin (2024) y qué parte no, y cómo se ven los datos
en cada caso. Código: `modelo_tesis/ll_estim.py` (`to_cells`, `cell_data`, `cell_logp`,
`score_parts`); el panel lo genera `modelo_tesis/gen_dataset.py`. Especificación:
`modelo_fin.md`, secs. 3-6.

---

## 0. Notación

**Estados.** Dos espacios, como en Gillingham:

- **X**, inicio de año: `act(j, a, w)` (coche activo de marca j, edad a = 1..24 y desgaste w),
  `term(j)` (el coche se murió: accidente o fin de vida) y `none` (sin coche).
- **H**, tenencia después de comerciar: `used(j, d, w)` (usado de edad d), `new(j)` y `none`.

Con J = 2 marcas, A = 25 y W = 71 puntos de w, hay n_act = 2 · 24 · 71 = 3,408 estados
activos y n = n_act + J + 1 = 3,411 estados en X (y en H).

**Dentro de un año**, quien empieza en x:

1. **Etapa 1** (escala σ): `keep`, `purge` (deshacerse y quedarse sin coche) o `trade`
   (deshacerse y comprar). Al deshacerse de un activo elige vender (μP − Ts) o chatarrear
   (escala σ_sell). Si hace trade, elige qué comprar: nuevo j o usado (j, d) y, dentro del
   nido, qué w (σ_w).
2. **Etapa 2** (escala σ_rep): si tiene un usado h de edad 1..23, decide reparar (r = 1) o
   no (r = 0). Un coche nuevo, la edad 24 y "sin coche" no reparan.
3. **Se maneja h**: se descompone con probabilidad s(h) (pasa a `term`); si sobrevive,
   w' = w + δ_j − acc_age_j − κ·r + η, η ~ N(0, σ_η²) discretizado en el grid, edad d + 1.

**Objetos que salen del equilibrio** (los precios P(j, a, w) nunca se observan; se resuelven
dentro de cada evaluación, régimen por régimen):

| símbolo | qué es | código |
|---|---|---|
| C_t,τ[x, o] | Pr(resultado o \| x), para el régimen t y el tipo τ | `outcome_probs` |
| buy_t,τ(h) | Pr(comprar h \| hace trade) (incluye la elección de w) | `c.buy` |
| p_t,τ(h) | Pr(reparar \| h) | `c.repair` |
| s(h) = s(j, d, w) | Pr(descomponerse) = logit⁻¹(acc_int_j + acc_age_j · d + w) | `breakdown_prob` |
| F_r(w' \| j, w) | transición de w dado r | `w_transition` |

Los **resultados observables** o son cinco: 0 keep, 1 purge-vende, 2 purge-chatarrea,
3 trade-vende, 4 trade-chatarrea. Un coche terminal siempre se chatarrea (o ∈ {2, 4}).

Dos propiedades del modelo hacen que la verosimilitud sea manejable:

- **s no depende de r** (la "opción 2" de `modelo_fin.md` sec. 3.3): reparar hoy no cambia la
  probabilidad de descomponerse hoy, solo el w' de mañana.
- **p(h) depende solo de h** (y del régimen y el tipo), no de cómo se llegó a h: el que se
  quedó el coche y el que lo acaba de comprar reparan con la misma probabilidad.

---

## 1. Los datos: panel y celdas

### 1.1 El panel (igual para los tres diseños)

`gen_dataset.simulate_panel`: por régimen t ∈ {ζ = −0.3, 0, +0.3}, N = 10,000 hogares
(repartidos entre los tipos pareja/soltero según f), K = 4 años, con el estado inicial
sacado de la distribución estacionaria del régimen. Una fila por hogar y año:

| columna | qué es |
|---|---|
| `regimen`, `tipo`, `id`, `k` | régimen (= el precio de reparación R_t que enfrenta), tipo de hogar, hogar, año |
| `x` | estado al inicio del año (índice en X: marca, edad y **w** del coche, o terminal, o sin coche) |
| `o` | resultado (0-4) |
| `h` | tenencia después de comerciar (índice en H, con su w si es usado) |
| `r` | reparó h este año (1/0; −1 si no podía) |

Lo que **siempre** se observa: régimen, tipo, x (incluido w), o, h (incluido el w del usado
comprado). Lo que **nunca**: precios de usados, choques de utilidad.

Ejemplo (ilustrativo) de un hogar:

| id | k | x | o | h | r |
|---|---|---|---|---|---|
| 17 | 0 | act(LB, 5, w = 0.30) | keep | used(LB, 5, 0.30) | 1 |
| 17 | 1 | act(LB, 6, w = 0.10) | keep | used(LB, 6, 0.10) | 0 |
| 17 | 2 | term(LB) | trade-chatarrea | used(HB, 3, −0.15) | 0 |
| 17 | 3 | act(HB, 4, w = 0.05) | purge-vende | none | −1 |

### 1.2 La unidad de observación: una transición h_{k−1} → (x_k, o_k, h_k)

La verosimilitud es **condicional en la tenencia del año anterior** h_{k−1}. Cada fila con
k ≥ 1 aporta la probabilidad de lo que pasó entre la tenencia de ayer y la de hoy:

    Pr(x_k, o_k, h_k | h_{k−1}, régimen, tipo)

Así, el año k = 0 de cada hogar solo sirve para dar h_0, y no se usa la distribución
inicial (no hay "condición inicial" en la verosimilitud). Por réplica: 3 regímenes × 10,000
hogares × 3 transiciones = **90,000 observaciones**.

En el diseño 1 la transición además lleva r_{k−1}: si h_{k−1} se reparó.

### 1.3 Celdas con conteos

Como todo es discreto, las observaciones idénticas se juntan en **celdas** con un conteo
(`to_cells`), y la log-verosimilitud es

    ℓ(θ) = Σ_celdas cnt_c · log Pr(celda c | θ, P(θ))

Las llaves de la celda son `(regimen, tipo, h_prev, r_prev, x, o, h)`, y cada diseño borra
información **antes** de agrupar:

| diseño | r_prev | x | celdas por réplica (media) |
|---|---|---|---|
| 1 | observado (0/1, o −1 si no podía reparar) | observado | 48,729 |
| 2 | borrado (−1 siempre) | observado | 45,802 |
| 3 | borrado | borrado (−1) si h_prev era un coche y o es chatarrear | 45,273 |

Agrupar es exacto (no hay pérdida de información respecto a la suma por observación): la
probabilidad de una observación depende solo de sus llaves. El régimen y el tipo son llaves
porque cambian los CCPs (R_t cambia p, el tipo cambia μ, u0, u1, Tb).

---

## 2. Los bloques comunes

Para una celda con h = h_{k−1}, régimen t y tipo τ (se omiten t, τ cuando no hay confusión):

**Transición física de h a x, condicional en r:**

    Pr(x | h, r) = s(h)                          si x = term(j)
                 = (1 − s(h)) · F_r(w' | j, w)   si x = act(j, d + 1, w')

con los casos especiales: un usado de edad 24 llega a `term` con probabilidad 1 (s = 1); un
nuevo tiene w = w₀ y r = 0; "sin coche" va a `none` con probabilidad 1 (ese factor es 1).

**Decisión del año k:**

    choice(x, o, h_k) = C[x, o] · buy(h_k)^{1(o es trade)}

C[x, o] da la probabilidad del resultado (keep / purge / trade × vender / chatarrear) y buy
la de qué coche se compró (marca, edad y w). Este bloque es idéntico en los tres diseños
cuando x se observa: los tres ven la decisión de comercio completa.

**Reparación:** p(h) = Pr(r = 1 | h), cero si h no puede repararse (`can_rep`).

---

## 3. Diseño 1: se observa r y el motivo de salida

### Qué se ve

Todo: x (con w'), o, h, y además r_{k−1}. Es la cota superior: el investigador ve la
decisión de reparar como si fuera un registro de talleres.

### Verosimilitud

    Pr(r, x, o, h_k | h) = Pr(r | h) · Pr(x | h, r) · C[x, o] · buy(h_k)^{trade}

con Pr(r | h) = p(h)^r (1 − p(h))^{1−r} si h puede repararse, y 1 si no.

**Es un logit dinámico estándar con dos decisiones observadas** (etapa 1 y etapa 2) y una
transición de estado observada. No hay mezcla. El logaritmo se separa en tres sumandos:

- log Pr(r | h): identifica σ_rep y la utilidad de reparar, sobre todo porque R_t cambia
  entre regímenes y r responde.
- log Pr(x | h, r): identifica κ (el efecto de reparar en w'), δ, σ_η (de las transiciones
  de w condicionales en r) y acc_int, acc_age (de los accidentes).
- log C + log buy: preferencias y costos de transacción (μ, u0, u1, Tb, Ts, σ_sell), como en
  Gillingham pero con w.

### Hu & Xin aquí

No se usa: la decisión se observa. El diseño 1 es la referencia contra la que se mide cuánto
cuesta no ver r (Hu & Xin hacen lo mismo en su sec. 5: comparan contra "choices observed").

---

## 4. Diseño 2: no se observa r (la pregunta central)

### Qué se ve

Igual que el diseño 1 menos r: x (con w'), o, h. Se sabe que el coche sobrevivió y con qué
desgaste llegó, pero no si se reparó.

### Verosimilitud

Como r no se ve, se **suma sobre r** (se integra la decisión no observada):

    Pr(x, o, h_k | h) = [ Σ_{r ∈ {0,1}} Pr(r | h) · Pr(x | h, r) ] · C[x, o] · buy(h_k)^{trade}

Para un coche que sobrevive esto es la mezcla

    f_t(w' | h) = (1 − s(h)) · [ (1 − p_t(h)) F_0(w' | j, w) + p_t(h) F_1(w' | j, w) ]

y para un accidente, s(h) (la mezcla desaparece porque s no depende de r).

En el código (`cell_logp`, rama `design != 1`): `trans = (1 − p1)·px[0] + p1·px[1]`.

### Hu & Xin aquí

Este es exactamente su problema: una decisión no observada (r) que afecta la transición de
un estado observado (w). Lo que se toma del paper:

1. **Identificación** (sec. 3 y sec. 6.3). La transición observada es una mezcla finita
   con pesos p_t(h) y componentes F_0, F_1:
   - *Assumption 2* (transición aditiva): w' = g(w, r) + η, con η independiente de r. Aquí
     g(w, r) = w + δ_j − acc_age_j − κ r y η ~ N(0, σ_η²).
   - *Assumption 7* (variable excluida): el precio de reparación R_t mueve p_t(h) pero no
     F_r. Por eso hay 3 regímenes: la variación de la mezcla entre regímenes, con las
     mismas componentes, separa F_0, F_1 y p_t(h) (ecs. 3.2-3.4). Hu & Xin necesitan 2
     valores; usamos 3.
   - *Observabilidad parcial* (sec. 6.3, ec. 6.4). La decisión tiene una parte observada
     (keep/purge/trade y qué se compra = a¹) y una no observada (reparar = a²). La
     transición observada se factoriza como f(x' | x, a¹) = Σ_{a²} f(x' | x, a) p(a² | x, a¹).
     En nuestro modelo p(a² | x, a¹) = p(h): la etapa 2 depende de la decisión de la
     etapa 1 solo a través de la tenencia h que resulta.
2. **Estimador** (sec. 4, ec. 4.1): máxima verosimilitud con la decisión no observada
   sumada **dentro** del logaritmo,

       ℓ = Σ_obs log Σ_a f(x_{t+1} − g(x_t, a) | ·) · Pr(a | x_t; θ)

   donde Pr(a | x; θ) sale de resolver el problema del agente. Nuestra verosimilitud es esa
   misma expresión con a = (a¹, a²) y a¹ observada.

Lo que **no** se toma del paper (y conviene decirlo en la tesis):

- **Sin sieve.** Hu & Xin estiman g y la distribución de η de forma no paramétrica (sieve
  MLE). Aquí g es lineal en r (un solo parámetro κ) y η es normal: es el caso paramétrico
  de su estimador. Con el DGP paramétrico del MC no hace falta el sieve; con datos reales
  sería una extensión natural (p. ej. κ por edad o por marca).
- **Sin primer paso no paramétrico.** Su Remark 2 propone un estimador en dos pasos
  (primero transición y CCPs por sieve, luego Hotz-Miller). Aquí todo es un solo paso de
  información completa. `modelo_fin.md` sec. 6 menciona contrastar contra el primer paso
  no paramétrico: **no está implementado**.
- **Equilibrio.** El paper es de agente único. Aquí los CCPs dependen de precios de
  equilibrio no observados, así que en cada evaluación se resuelve el equilibrio de cada
  régimen (punto fijo anidado: `LLEval` con pasos de cuerda). Hu & Xin no cubren esto; la
  identificación de la mezcla no cambia porque los precios son función de θ y del régimen,
  no de r.

---

## 5. Diseño 3: no se observa r ni el motivo de salida (como el registro danés)

### Qué se ve

Como el diseño 2, salvo cuando **un coche sale del parque**: si el hogar tenía un coche
(h_{k−1} ≠ none) y el resultado es chatarrear (o ∈ {2, 4}), no se sabe si fue

- un **accidente** (x = term: el coche se murió solo y se chatarrea por fuerza), o
- un **chatarreo voluntario** de un coche que seguía vivo (x = act(j, d + 1, w') y el dueño
  eligió chatarrear en vez de vender).

En esas filas x se borra (y con él el w' de un coche que habría sobrevivido). En
`to_cells`: `d.loc[salida, "x"] = -1`. Es lo que ve Gillingham en el registro danés
(su apéndice D).

### Verosimilitud

Filas sin salida: igual que el diseño 2. Filas con salida, se suma sobre r **y** sobre x:

    Pr(o, h_k | h) = Σ_r Pr(r | h) · [ s(h) · C[term(j), o]
                                      + (1 − s(h)) · Σ_{w'} F_r(w' | j, w) · C[act(j, d + 1, w'), o] ]
                     · buy(h_k)^{trade}

- Primer término: accidente; C[term, o] es la probabilidad de purge vs trade desde la
  terminal (chatarrear es forzoso).
- Segundo término: sobrevive con algún w' no observado y el dueño elige chatarrearlo.
  La probabilidad de chatarrear voluntariamente depende de w' (un coche más gastado vale
  menos), así que la mezcla sobre r también entra aquí.

En el código: la rama `design == 3` de `cell_logp` (`mix`, `lp_exit`).

### Hu & Xin aquí

Lo mismo que en el diseño 2 para la mezcla sobre r. La segunda mezcla (accidente vs
chatarreo) **no** es de Hu & Xin: es la verosimilitud "parcial" del apéndice D de
Gillingham. Las dos mezclas no se cruzan porque s depende del w de inicio de año (observado
en h) y no de r. Por eso el término del accidente sale fuera de la suma sobre r.

---

## 6. Para comparar: la verosimilitud de Gillingham sobre los mismos datos

`gillingham/ll_estim.py`, estimada sobre el mismo panel (`montecarlo.to_gillingham`):

- **Datos:** se borra w (el estado es (j, a)) y r; tampoco se usa el régimen (en Gillingham
  R no entra, así que los tres regímenes se juntan). Celdas `(tipo, h_prev, o, h)` (parcial)
  o `(tipo, h_prev, x, o, h)` (completa): ~1,065 y ~1,108 celdas por réplica.
- **"Parcial"** (la del paper, como el diseño 3): Pr(o, h_k | h) = (Q C)[h, o] · buy(h_k)^{trade}.
  Integra el accidente pero no hay desgaste ni reparación.
- **"Completa"** (como el diseño 2, pero sin w): Pr(x, o, h_k | h) = Q[h, x] · C[x, o] · buy^{trade}.

Como el modelo está mal especificado (no tiene w ni reparación), sus estimaciones están
sesgadas (`reporte_mc.md`, sec. 2).

---

## 7. Cómo se maximiza (lo mismo en los tres diseños)

- **Punto fijo anidado.** Cada evaluación de ℓ(θ) necesita el equilibrio z_t(θ) = (EV de
  cada tipo, P) de los 3 regímenes. Se resuelve con pasos de cuerda desde el equilibrio
  anterior, con el predictor z + (dz/dθ) Δθ (`LLEval`, `equilibrio.solve_chord`).
- **Gradiente implícito.** dz_t/dθ = −F_z⁻¹ F_θ (`dz_dx`). El score de cada celda incluye
  el efecto directo de θ y el indirecto vía los precios.
- **Optimizador.** L-BFGS hasta acercarse; luego BHHH arrancando en la mejor evaluación,
  hasta g'B⁻¹g/N < 1e-9.
- **Errores estándar.** Inversa de la matriz BHHH Σ_c cnt_c s_c s_c' (producto exterior de
  los scores por celda; una celda con conteo n cuenta como n observaciones idénticas), con
  el método delta para pasar a la escala natural (μ, κ, σ_η, σ_rep se estiman en logaritmo;
  σ_sell con un sigmoide).
- **Parámetros libres:** 27 (μ, u0, u1, Tb, Tb sin coche por tipo/marca; Ts, Ts inspección,
  σ_sell; acc_int, acc_age, δ por marca; u_w, κ, σ_η, σ_rep). Fijos: σ, σ_trade, σ_w, β,
  p_new, p_scrap, u2 y R (dato).

---

## 8. Resumen

| | diseño 1 | diseño 2 | diseño 3 |
|---|---|---|---|
| observa r | sí | no | no |
| observa x cuando el coche sale | sí | sí | no |
| celdas | (régimen, tipo, h_prev, r, x, o, h) | (régimen, tipo, h_prev, x, o, h) | ídem, con x = −1 en las salidas |
| mezcla sobre r | no | sí (en w') | sí (en w' y en el chatarreo) |
| mezcla accidente/chatarreo | no | no | sí |
| papel de Hu & Xin | referencia ("choices observed") | identificación (Assumptions 2 y 7, ec. 6.4) y estimador (ec. 4.1, paramétrico) | igual que el diseño 2 + apéndice D de Gillingham |
