# Verosimilitud estructural de modelo_fin (segunda etapa)

Qué se estima, con qué datos, cómo se escribe la verosimilitud, cómo se evalúa y qué
identifica cada parámetro. Es el DNFXP de Gillingham (sec. 5.1 y apéndice D; réplica en
`gillingham.md`) con la mezcla sobre la reparación no observada de Hu & Xin adentro.

---

## 1. Qué hay hoy y qué falta

`modelo_fin/loglikelihood.py` es la **primera etapa** de Hu & Xin. Usa solo las
transiciones de s de los coches que sobreviven:

    f(s' | h, t) = (1 − p_t(h)) F_0(s' | h) + p_t(h) F_1(s' | h)

Recupera la transición F_r (s_repair, s_sigma, drift) y las probabilidades de reparar
p_t(h), **sin decir de dónde salen esas p**. Las p son libres (o un logit en R).

La **segunda etapa** (lo que falta) explica todo con el modelo: las p_t(h), las
decisiones keep / trade / purge, qué coche se compra, en función de los parámetros de
preferencias y costos θ, y del equilibrio.

## 2. Parámetros

**Se estiman (θ):**

| grupo | parámetros | por tipo de hogar |
|---|---|---|
| dinero | mu | sí |
| utilidad de tener coche | u0_j, u1_j | sí |
| costos del comprador | tc_buy, tc_buy_nocar | sí |
| costos del vendedor | tc_sell, tc_sell_inspect | no |
| riesgo | u_s (desutilidad de un coche frágil) | no |
| reparación | sigma_repair | no |
| dinámica de ℓ = logit(s) | acc_int_j, acc_age_j, s_repair, s_sigma | no |
| chatarreo (si se agrega) | sigma_sell | no |

**Fijos (conocidos o normalizados):** beta, p_new, p_scrap, sigma = 1 (escala del
shock), u_none = 0. **Datos:** R_t(j, a), el precio de reparación de cada año t.

Los parámetros de la dinámica de ℓ se pueden estimar en la primera etapa y fijarse en la
segunda (dos etapas) o estimarse junto con todo (una etapa). Ver sección 6.

## 3. Datos (los del Monte Carlo)

Panel de hogares i = 1..N, años t = 1..K. En cada año se observa:

| variable | qué es | ¿se observa? |
|---|---|---|
| τ_i | tipo de hogar | sí |
| x_it | estado: (j, a, ℓ) o "sin coche" o terminal | sí |
| o_it | keep / purge / trade | sí |
| h_it | tenencia: el mismo coche, el comprado (j, d, ℓ) o nuevo j, o ninguno | sí |
| r_it | si reparó | **no** |
| x_i,t+1 | estado del año siguiente | sí |
| P | precios de usados | no (base); opcional |
| R_t | precios de reparación del año | sí |

Accidente contra salida: si el coche pasa a terminal, se ve que salió. Si el modelo
tiene chatarreo endógeno, no se sabe si fue accidente o chatarreo (como en Gillingham).

## 4. Probabilidad de una observación

Un año de un hogar de tipo τ en el año-régimen t se descompone en dos partes que
corresponden a la secuencia del modelo:

    inicio x  --(etapa 1: o, h)-->  tenencia h  --(etapa 2: r, no se ve)-->  manejas  --(azar)-->  x'

    Pr(o, h, x' | x)  =  Pr(o, h | x)   ×   Pr(x' | h)
                         [elección]          [transición con la r integrada]

### Parte 1: la elección (etapa 1)

    Pr(keep  | x) = keep_τ,t(x)
    Pr(purge | x) = purge_τ,t(x)
    Pr(trade, h | x) = trade_τ,t(x) · buy_τ,t(h)

Son las CCPs logit de siempre (`probabilities.py`). Dependen de θ directamente
(utilidades, costos) y también de EV_τ,t y P_t, que salen del equilibrio.

### Parte 2: la transición, con r integrada

r no se ve, así que se suma sobre sus dos valores. La probabilidad de reparar es la CCP
de la etapa 2:

    p_τ,t(h) = Pr(r = 1 | h) = Λ( [W_1(h) − W_0(h) − mu_τ R_t(h)] / sigma_repair )

(Λ = logística; W_r(h) = valor de continuación si reparas o no, que sale de EV_τ,t.)

    Pr(x' | h) = (1 − p_τ,t(h)) Q_0(h, x') + p_τ,t(h) Q_1(h, x')

Con la opción 2:
- **Sobrevive** (prob 1 − s(h), igual con o sin reparar):
  Q_r(h, (j, d+1, ℓ')) = (1 − s(h)) · F_r(ℓ' | h), con F_r normal discretizada de media
  ℓ + acc_age_j − s_repair · r.
- **Accidente** (prob s(h)): no depende de r, así que no informa sobre r.

### La log-verosimilitud

    LL(θ) = sum_{i,t} [ log Pr(o_it, h_it | x_it; θ)
                       + log( (1 − p(h_it)) F_0(ℓ_i,t+1 | h_it) + p(h_it) F_1(ℓ_i,t+1 | h_it) )  si sobrevive
                       + log s(h_it)  o  log(1 − s(h_it))  (salida o supervivencia) ]

La segunda línea es la misma mezcla de la primera etapa, **pero ahora p no es libre: es
la CCP del modelo**. Eso es lo que ata la reparación no observada al resto del modelo.

En la práctica se agrega a celdas con conteos (τ, t, x, o, h, x'), como en Gillingham
(ec. 40), y se suman conteos × log probabilidades.

## 5. Cómo se evalúa LL(θ) (lo que hace la computadora)

Para cada θ que propone el optimizador:

1. **Resolver los equilibrios.** Para cada año-régimen t (cada vector R_t):

       z_t = (EV_0,t, EV_1,t, P_t)    tal que   F(z_t, θ, R_t) = 0

   con Newton-Krylov (`newton_krylov.md`). Cada año es un equilibrio estacionario
   distinto (la lectura de "años" que está pendiente; ver pendiente 10 de v1.0).
2. **Calcular las CCPs** de la etapa 1 y de la etapa 2 en (z_t, θ).
3. **Sumar** conteos × log probabilidades.

Esto es el "anidado" del nested fixed point: un ciclo externo sobre θ y un punto fijo
interno (el equilibrio) en cada evaluación. Si P se dejara fijo, sería estimar un modelo
con precios equivocados: cuando θ cambia, el mercado se reacomoda y P cambia.

## 6. Una etapa contra dos etapas

**Una etapa (máxima verosimilitud completa).** Se maximiza LL(θ) con todo libre,
incluida la dinámica de ℓ. Es la más eficiente y la que se recomienda como resultado
principal.

**Dos etapas (Hu & Xin y luego estructural).**
1. Primera etapa (`loglikelihood.py`, ya existe): estimar F_r y p_t(h) solo con las
   transiciones de ℓ.
2. Segunda etapa: fijar la dinámica de ℓ en lo estimado y maximizar LL(θ) sobre el resto.

Sirve para:
- dar valores iniciales a la de una etapa;
- mostrar el argumento de Hu & Xin paso por paso (primero la mezcla sin modelo, luego
  el modelo);
- un chequeo de especificación: las p̂_t(h) libres de la primera etapa contra las
  p_t(h; θ̂) del modelo.

Los errores estándar de la segunda etapa hay que corregirlos por la primera (Murphy-
Topel), o sacarlos por bootstrap.

## 7. El gradiente: derivar a través del equilibrio

LL depende de θ directamente y a través de z(θ):

    dLL/dθ = ∂LL/∂θ  +  ∂LL/∂z · dz/dθ

El término difícil es dz/dθ: cuánto se mueve el equilibrio si cambia θ. Como
F(z(θ), θ) = 0 para todo θ, derivando:

    F_z · dz/dθ + F_θ = 0   =>   dz/dθ = −F_z⁻¹ F_θ        (función implícita)

No hace falta derivar a través de las iteraciones del solver: solo hace falta la
solución. Dos formas de usarlo:

- **Adjunta (para el gradiente total):** resolver una sola vez F_z^T λ = (∂LL/∂z)^T y
  luego dLL/dθ = ∂LL/∂θ − λ^T F_θ. Un sistema lineal por año, sin importar cuántos
  parámetros haya. Sirve para L-BFGS.
- **Directa (para los scores por celda):** BHHH y los errores estándar necesitan el
  gradiente de cada celda, así que hace falta dz/dθ completo: un sistema lineal por
  parámetro (~30) y por año. En GPU es barato; se usa solo en la fase BHHH y al final.

Los dos sistemas se resuelven con GMRES y productos jvp/vjp, sin armar F_z.

**Prueba obligatoria:** gradiente analítico contra diferencias finitas a tamaño chico
(en la réplica de Gillingham coincidió a 5e-8).

## 8. Qué identifica a cada parámetro

| parámetro | qué lo identifica |
|---|---|
| acc_int, acc_age, s_repair, s_sigma | la mezcla de transiciones de ℓ (Hu & Xin) y las tasas de salida |
| sigma_repair y mu | **la respuesta de la mezcla a R_t**: si sube R_t y baja p_t(h), el modelo lee cuánto pesa el dinero frente al shock de reparar. Esta es una fuente de identificación de mu que Gillingham no tiene (él usa p_new) |
| u0, u1 | keep / trade / purge por edad y la elección de edad al comprar |
| u_s | cómo cae keep con ℓ dentro de (j, a) |
| tc_buy contra tc_sell | **dos tipos con mu distinta** (v1.4, `gillingham.md`). Con un solo tipo, sin precios, no se separan |
| tc_buy_nocar | compras desde "sin coche" (estado observado) |
| tc_sell_inspect | el zig-zag en edades pares |
| sigma_sell (si hay chatarreo) | el reparto de salidas entre accidente y chatarreo, por forma funcional (como en Gillingham) |

**Por qué R ayuda tanto:** R_t mueve p_t(h) (la CCP de reparar) pero no la transición
F_r (Assumption 7 de Hu & Xin). En la primera etapa eso separa p de F; en la segunda,
da variación exógena del costo de una decisión con dinero, que pega directo en mu.

## 9. Variantes

- **Información asimétrica** (`info_asimetrica.md`): Pr(trade, h = (j, d, ℓ) | x) =
  trade(x) · buy(j, d) · π(ℓ | j, d). π sale del equilibrio, como P.
- **Precios observados:** se agrega log f(P_obs | P(θ)) con error de medición. Ancla
  tc_buy y tc_sell aunque haya un solo tipo.
- **Datos tipo daneses (sin ℓ):** ℓ es latente y hay que integrarla con un filtro
  (forward algorithm) sobre la historia de cada coche. Mucho más caro; queda fuera de
  este plan.

## 10. Pseudocódigo

```
def loglik(x_free, datos):
    θ = de_libre(x_free)                       # exp para mu, sigma; resto igual
    LL = 0
    for t in años:                             # vmap sobre años en GPU
        z_t = equilibrio(θ, R_t, z_init=z_previo[t])     # Newton-Krylov, jit
        ccp1, p_rep = ccps(z_t, θ, R_t)                 # etapas 1 y 2
        LL += sum(conteos_t * log ccp1[celdas_t])
        LL += sum(conteos_sobrevive_t * log((1 - p_rep) F0 + p_rep F1))
        LL += sum(conteos_salida_t * log s + conteos_sigue_t * log(1 - s))
    return LL

# gradiente: jax.custom_vjp sobre equilibrio() con la regla implícita (adjunta)
# optimización: L-BFGS (gradiente adjunto) -> BHHH (scores por celda) -> se
# varios arranques; guardar TODOS (pendiente 4 de la lista)
```

## 11. Dónde puede fallar

1. **Máximos locales** (ya pasó en la réplica). Varios arranques y guardar todos.
2. **Equilibrio que no converge** para θ lejos de la verdad. Hay que devolver LL = −∞ y
   que el optimizador retroceda. Ayuda arrancar el equilibrio desde el z del θ anterior.
3. **mu y sigma_repair** se identifican por su cociente en la etapa 2; los separa el
   resto del modelo (p_new en la compra de nuevos). Revisar la correlación entre
   réplicas en el MC.
4. **Varios equilibrios** (sobre todo con información asimétrica): la LL puede saltar si
   el solver cambia de equilibrio entre evaluaciones. Arrancar siempre desde el z
   anterior.
