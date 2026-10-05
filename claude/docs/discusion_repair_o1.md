# Opción 1 vs. opción 2 para reparar: ¿qué permiten los datos tipo Gillingham?

Fuentes: Gillingham, Iskhakov, Munk-Nielsen, Rust & Schjerning (2019, WP "Equilibrium
Trade in Automobiles", en `po/Fuentes/01 Empirical/BLP y Dynamic/BLP Demanda/`) y
Hu & Xin (2024, J. Econometrics, `Tesis Carpeta Final/unobs_choice.pdf`).

## Las dos opciones

- **Opción 2 (implementada).** Reparar solo mueve la media de s'. La probabilidad de
  descomponerse este periodo es el s actual, se repare o no:

      Pr(term | h, r) = s,     s' = m(j, d, s) - s_repair * r + eta.

- **Opción 1 (`opcion_1_reparar.md`).** Reparar baja ya la probabilidad de
  descomponerse este periodo, s_eff(r, s) = s - s_repair * r, y s' parte de s_eff.

## Qué hay en los datos de Gillingham

De la sección 6 y el apéndice F.1:

1. **Registros administrativos daneses 1996–2008**, anuales, de todos los hogares y todos
   los coches de particulares. Se observa quién tiene qué coche (tipo j, edad a), las
   transiciones de propiedad (keep, trade, purge) y el coche que se compra.
2. **No hay precios del mercado secundario.** Los P son una salida del modelo, no un dato.
3. **No hay accidentes.** Solo se observa el "scrappage", que definen como el fin de un
   periodo de propiedad sin que empiece otro en 3 años. Ese scrappage mezcla accidentes y
   chatarreo voluntario, y separarlos es justo lo que el modelo estructural tiene que
   hacer (pp. 36 y 48).
4. **No hay reparaciones ni una variable como s.** La probabilidad de accidente es
   una función paramétrica de (j, a), un logit binario (Tabla 4), no un estado
   observado.
5. **Inspecciones obligatorias (syn) bianuales a partir de los 4 años.** De ellas sale el
   odómetro (la regresión de manejo, Tabla 6). Si la inspección encuentra fallas, el dueño
   **tiene que repararlas** para seguir manejando. Esto produce el "zig-zag" del
   scrappage en edades pares, que modelan con un dummy de edad par en la utilidad y con un
   costo de venta más alto en años de inspección (Tabla 5: 0.3454 vs 0.9106).

Conclusión directa: **los datos de Gillingham no traen s ni reparaciones**. La pregunta
real no es qué opción encaja con su base, sino qué datos tendrías que tener (o simular en
el Monte Carlo) para que cada opción esté identificada.

## Identificación con Hu & Xin

### Opción 2

Para un usado h que sobrevive:

    f(s' | h, sobrevive) = sum_r  p(r | h) * phi( s' - m(h) + s_repair * r )

Es exactamente la ec. 6.4 de Hu & Xin con la Assumption 2 (s' = m(r, s) + eta, eta
independiente de s): una mezcla de dos componentes desplazados, con pesos iguales a las
CCPs. Que el coche sobreviva **no depende de r**, así que condicionar en sobrevivir no
sesga los pesos. El precio de reparación R(j, a) es la variable excluida (Assumption 7):
mueve p(r | h) pero no phi. **Encaja directo con el paper.**

Solo necesitas observar s_t y s_{t+1} en años consecutivos para el mismo coche, más
keep/trade.

### Opción 1

Ahora sobrevivir sí depende de r. La densidad de s' entre los que sobreviven es

    f(s' | h, sobrevive) ∝ sum_r  p(r | h) * (1 - s_eff(r, s)) * phi( s' - m_r(h) ).

1. **Los pesos ya no son las CCPs** sino p(r|h)(1 - s_eff(r,s)), normalizados. Hay
   selección por supervivencia: los reparados sobreviven más y por eso están
   sobrerrepresentados entre los s' observados. Los componentes phi(· - m_r) se siguen
   identificando con el argumento de Hu & Xin (familia de localización con el mismo eta),
   y con ellos los pesos. Para recuperar p(r|h) hay que deshacer el factor (1 - s_eff),
   que depende de s_repair; s_repair a su vez sale de la diferencia de medias
   m_0 - m_1. Es identificable, pero ya no es la ec. 6.4 tal cual: es una extensión que
   tendrías que escribir y probar.
2. **Un momento adicional, si se observan las descomposturas:**

       Pr(term | h) = sum_r p(r | h) * s_eff(r, s).

   Con s_persist = 1, s_repair aparece a la vez en la media y en s_eff, y eso da una
   restricción de sobreidentificación.
3. **Pero en datos tipo Gillingham no se observan las descomposturas**, solo el
   scrappage total. Además el modelo de la tesis ya no tiene chatarreo endógeno: ahí todo
   scrappage antes de a_max es un accidente y el momento sí se observa en el Monte Carlo.
   En datos daneses reales esto no se cumpliría, porque el zig-zag muestra mucho
   chatarreo voluntario.

### Cuándo se mide s

La opción 1 solo tiene sentido si el s_t observado es **anterior** a la reparación. El
ejemplo natural es la inspección danesa: se miden las fallas, luego se decide reparar y
la reparación protege ese mismo año. Si s se mide **después** de reparar (p. ej. "pasó la
inspección"), el s observado ya incorpora r. Entonces r deja de ser una decisión no
observada que mueve la transición y pasa a ser parte del estado, y el planteamiento de
Hu & Xin cambia.

Las inspecciones danesas tienen además dos problemas para cualquiera de las dos opciones:

- **Son bianuales.** Solo se observaría (s_t, s_{t+2}), que mezcla dos decisiones de
  reparar: 4 componentes con pesos p(r_t) p(r_{t+1} | s_{t+1}). Hu & Xin necesitan
  transiciones de un periodo.
- **Solo desde los 4 años.** Con a_max = 7 apenas habría dos inspecciones por coche.

## Recomendación

**Quedarse con la opción 2 como caso base**, por tres razones:

1. Es literalmente la estructura de Hu & Xin (Assumption 2 + ec. 6.4 + Assumption 7). El
   Monte Carlo mide el desempeño de su estimador, no el de una extensión propia.
2. La ventaja de la opción 1 (el momento de descomposturas) requiere observar
   accidentes, que justo es lo que la base de Gillingham no tiene.
3. Con la decisión secuencial nueva (`estructura_decision.md`), en la opción 2 los pesos
   de la mezcla son p(repair | h) para keepers y compradores por igual. En la opción 1
   habría que corregir la selección también para los compradores.

**La opción 1 queda como ejercicio de robustez:** simular datos con la opción 1 y
estimar como si fuera la opción 2, para medir el sesgo por especificar mal el timing.
Eso responde a un evaluador que pregunte "¿y si reparar protege ya este año?".

## Una idea que sale de los datos daneses

Las inspecciones obligatorias en edades pares parecen una fuente natural de variación en
los costos de reparar: en años de inspección reparar es casi obligatorio. Pero **no sirve
como variable excluida** en el sentido de la Assumption 7. Si la inspección obliga a
reparar, también cambia la transición de s (vía r) y probablemente la utilidad (el dummy
de edad par de Gillingham). El candidato limpio sigue siendo R(j, a) variando entre
años, como ya está en el modelo.
