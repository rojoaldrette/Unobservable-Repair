# Información asimétrica: un precio por (j, a) y s descubierta al comprar

Pregunta: si el comprador de un usado no ve s (la ve solo después de comprar) y hay un
solo precio P(j, a), ¿bajan las incógnitas del equilibrio? ¿Cómo cambian las
matemáticas?

Respuesta corta: **baja el número de precios (de 7,200 a 72 con a_max = 25), pero no el
número total de incógnitas**. Aparecen las creencias del comprador sobre s, que son
otro punto fijo (expectativas racionales, tipo Akerlof). El sistema tiene más o menos el
mismo tamaño. Lo que sí mejora es su forma, su realismo y la historia de la tesis.

Notación como en `modelo_fin.md`: x = (j, a, s) estado del dueño, h = tenencia
post-comercio, S = n_s, A = a_max. Con a_max = 25, J = 3 y n_s = 100:
J(A−1) = 72 celdas (j, a) y J(A−1)S = 7,200 celdas (j, a, s).

---

## 1. Modelo actual (información simétrica)

- El comprador elige h = (j, d, s) con s visible. Precio P(j, d, s).
- Nido de s al comprar (sigma_s = 0.1): el comprador elige s casi por valor neto.
- Equilibrio: z = (EV, P) con

      EV = T(EV, P)                               n ecuaciones
      log D(j,d,s) − log S(j,d,s) = 0             J(A−1)S ecuaciones

- Problemas conocidos: celdas (j, d, s) sin masa con precios sin significado
  (pendiente 6 de v1.0) y la dependencia del grid que obligó a meter el nido sigma_s.

## 2. Modelo con información asimétrica

### Supuestos

1. El dueño conoce s de su coche (lo ha manejado).
2. El comprador solo ve (j, d). Al comprar recibe un coche al azar del "pool" de coches
   ofrecidos de ese (j, d). Descubre s **justo después de comprar**, antes de la etapa 2
   (reparar o no).
3. Un solo precio P(j, d) por celda (j, d).
4. Creencias del comprador: π(s | j, d) = distribución de s entre los coches ofrecidos.
   Expectativas racionales: π coincide con la composición real de la oferta.

### Lo que cambia en las ecuaciones

**Vendedor.** Deshacerse del coche paga lo mismo para todo s:

    disposal(j, a, s) = mu P(j, a) − Ts(a)          (antes: mu P(j, a, s) − Ts(a))

El valor de quedarse sí depende de s (baja con s), así que **los dueños de coches malos
venden más**: selección adversa.

**Comprador.** Compra una lotería sobre s:

    v_buy(j, d) = sum_s π(s | j, d) W(j, d, s) − mu P(j, d) − Tb

donde W(j, d, s) es el valor de tener h = (j, d, s) **ya con la etapa 2 incluida**
(emax de reparar o no). Como descubre s antes de reparar, la opción de reparar tiene
valor de seguro contra comprar un limón.

El nido sigma_s desaparece: el comprador ya no elige entre celdas de s.

**Creencias (el nuevo punto fijo).**

    π(s | j, d) = q(j, d, s) (1 − keep(j, d, s))  /  sum_s' q(j, d, s') (1 − keep(j, d, s'))

π depende de q y de las CCPs. Pero q depende de π, porque los compradores terminan en
h = (j, d, s) con probabilidad buy(j, d) · π(s | j, d). Por eso es un punto fijo, no una
fórmula.

**Matriz de comercio.** La fila de trade se vuelve

    Ω_trade(x -> (j, d, s)) = trade(x) · buy(j, d) · π(s | j, d)

**Mercado.** Se vacía por (j, d), no por (j, d, s):

    D(j, d) = trade_mass · buy(j, d)
    S(j, d) = sum_s q(j, d, s) (1 − keep(j, d, s))

Si π es consistente, la composición de lo comprado es igual a la de lo ofrecido, así que
no hace falta una condición por s.

### Equilibrio

    z = (EV, P, π)

    EV = T(EV, P, π)                      n ecuaciones           (EV sigue con s: el dueño la ve)
    log D(j,d) − log S(j,d) = 0           J(A−1) ecuaciones
    π = Φ(EV, P, π)                       J(A−1)(S−1) ecuaciones (cada π suma 1)

### Conteo de incógnitas (a_max = 25, n_s = 100, J = 3)

| | simétrica | asimétrica |
|---|---|---|
| EV por tipo de hogar | 7,204 | 7,204 |
| precios | 7,200 | **72** |
| creencias | 0 | 7,128 |
| total con 2 tipos | 21,608 | 21,608 |

Mismo tamaño. Las creencias se pueden eliminar como incógnitas si se itera sobre ellas
(dado π, resolver (EV, P); actualizar π <- Φ). Es un ciclo anidado más lento y no
necesariamente estable. Con Newton-Krylov conjunto sobre z = (EV, P, π) el costo es
parecido al del modelo actual (ver `newton_krylov.md`).

## 3. Existencia, unicidad, desenredo del mercado

- **Akerlof sin shocks:** el mercado de una celda (j, d) se puede "desenredar" (solo se
  ofrecen limones, el precio cae, salen los buenos...) hasta no tener comercio.
- **Con shocks logit** siempre hay algo de oferta de cada s (hay quien vende por un
  shock de gusto), así que π está bien definida y el desenredo total no ocurre. El
  logit suaviza el problema; esto ayuda a la existencia.
- **Unicidad no está garantizada.** Puede haber un equilibrio de precio alto / buena
  calidad y uno de precio bajo / limones. Newton encuentra uno según el arranque.
  Habría que probar varios arranques de P y π (sobre todo π = composición de q contra
  π = distribución de s de los keepers).

## 4. Beneficios

1. **Precios con sentido en todas las celdas.** Se acaban las celdas (j, d, s) sin masa
   con precios arbitrarios: toda celda (j, d) con coches tiene oferta.
2. **Sin el nido sigma_s.** Era un parche para la dependencia del grid. Con un solo
   precio por (j, d), el número de alternativas del comprador no depende de n_s.
3. **Realismo y datos.** Los precios de usados que se pueden conseguir (anuncios, guías
   de precios) vienen por modelo y edad, no por s. Si algún día se usan precios como
   dato (lo que identifica tc_buy y tc_sell con un tipo, ver `gillingham.md`), un precio
   por (j, d) es lo que se observa.
4. **Encaja con la pregunta de la tesis.** La reparación tampoco la ve el comprador.
   - Quien piensa vender no recupera lo que gasta en reparar, así que repara menos
     antes de vender (riesgo moral).
   - Quien acaba de comprar un usado y descubre un s alto repara más.
   - Predicción: Pr(repair) más alta justo después de comprar un usado y más baja antes
     de vender. Se puede contrastar con el estimador de Hu & Xin.
5. **Conexión con la inspección (syn).** Si la inspección de las edades pares revela s,
   la asimetría desaparece en años de inspección. Es otra explicación posible del
   zig-zag de Gillingham.

## 5. Costos

1. **No baja el tamaño del sistema** (ver la tabla del conteo).
2. **Posible multiplicidad** de equilibrios (sección 3).
3. **Tensión con Hu & Xin.** El estimador supone que el econometrista ve s. Si los
   agentes no la ven antes de comprar, hay que justificar de dónde la saca el
   econometrista (p. ej. registros de inspección que el comprador no consulta). En el
   Monte Carlo no es problema.
4. **Deja de anidar a Gillingham de forma trivial.** Sin s (sigma de eta -> 0 y sin
   reparar), todos los coches de (j, d) tienen el mismo s y la asimetría no importa: sí
   lo anida, pero solo en ese límite.

## 6. Comentario

Lo recomendable es implementarlo como **opción** (`info = "sym" | "asym"`) cuando el
modelo base con log-odds, dos tipos y a_max = 25 ya funcione, no antes. La comparación
entre los dos equilibrios (mismos parámetros) da un resultado propio de la tesis: cuánto
descuento por limones hay en P(j, d) y cómo cambia la reparación por edad cuando el
comprador no la ve. La maquinaria (Newton-Krylov conjunto, gradiente implícito) es la
misma: solo se agrega el bloque π a z.

En la verosimilitud estructural (`verosimilitud_estructural.md`, sec. 8) el único cambio
es que la probabilidad de comprar h = (j, d, s) es buy(j, d) · π(s | j, d).
