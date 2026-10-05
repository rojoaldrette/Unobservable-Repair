# Inspecciones, zig-zag del scrappage y reparación (ideas A y B)

Estado: **propuesta, no implementada.** El código actual (`claude/scripts/modelo_fin/`) no tiene
inspecciones.

## El hecho en Gillingham

- **Qué se observa.** En los registros daneses el scrappage es mucho mayor en edades
  pares >= 4. Es el "zig-zag", y los autores verificaron que no es un artefacto de
  medición (sec. 6.2).
- **Por qué pasa.** La inspección es bianual desde los 4 años. Si encuentra fallas, hay
  que repararlas para seguir circulando, y eso vale tanto si te quedas el coche como si
  lo vendes.
- **Cómo lo modelan.** Con un dummy de edad par en la utilidad (de -15% a -50% de la
  utilidad de un coche nuevo) y un costo de venta más alto en años de inspección
  (Tabla 5: 2.1929 contra 0.9106; ver `gillingham.md`).
- **Su interpretación** (introducción): "most Danes prefer to scrap their vehicles
  rather than incur the time and expense to repair their vehicles to pass inspection".

Lectura para la tesis: el scrappage endógeno de Gillingham en años de inspección es, en
el fondo, una **decisión de no reparar** que su modelo no tiene. El zig-zag es evidencia
de que el margen de reparación existe.

Vender en lugar de reparar en años de inspección **no** genera el zig-zag, por dos
razones:

1. Vender no saca el coche del parque: lo sigue manejando el comprador.
2. La inspección también obliga al comprador, así que vender no evita reparar.

## Notación común

- **Edades de inspección:** I = {a par, a >= inspect_age_min}. Con a_max = 7 son
  a = 4 y 6.
- **Prob. de reprobar:** kappa(s), creciente en s; por ejemplo
  `kappa(s) = 1 / (1 + exp(-(k0 + k1 * s)))`, con k0 y k1 como calibración propia.
- **Timing:** la inspección ocurre **al inicio** del periodo en que el coche cumple una
  edad en I, antes de comerciar. Se evalúa con el s de ese momento, s_{t+1}, que ya
  incorpora la reparación del año anterior. Esto es compatible con la opción 2 sin
  tocar el timing de la reparación.

---

## Idea A: reprobar es una salida exógena

Si el coche reprueba, se da de baja: pasa a terminal y el dueño cobra p_scrap. Nadie
decide sacar un coche; solo la mecánica y la inspección lo hacen.

### Cambios al modelo

Solo cambia Q (de H a X). Para un usado h = (j, d, s) cuya edad siguiente d+1 está en I:

    Pr(act(j, d+1, s') | h, r) = (1 - s) * F_r(s' | h) * (1 - kappa(s'))
    Pr(term(j)       | h, r) = s + (1 - s) * sum_{s'} F_r(s' | h) * kappa(s')

Para las demás edades, Q queda igual. En código: una máscara por edad en
`physical_matrices` y el factor (1 - kappa(s')) en las columnas de destino. Q_0 y Q_1
siguen siendo matrices fijas (no dependen de EV ni de P), así que:

- `continuation_values` debe usar el mismo factor (la Bellman ve el riesgo de reprobar);
- el jacobiano dT/dEV = beta M sigue valiendo;
- M = Ω Q no cambia de forma.

### Implicaciones económicas

1. **Zig-zag en las salidas.** El scrappage observado (accidentes + reprobados) sube en
   edades pares sin ninguna decisión de chatarrear.
2. **La reparación se vuelve preventiva.** En el año anterior a una inspección (edades
   impares 3 y 5), reparar baja el s' con el que se llega a la inspección. Por eso
   Pr(repair | h) debería subir en esos años: zig-zag en las CCPs de reparar, ahora por
   anticipación.
3. **Precios.** Los coches que están por inspeccionarse (y con s alto) valen menos: el
   comprador hereda el riesgo de reprobar. Esto genera zig-zag en P, que Gillingham
   reporta (fig. 7b) y obtiene con el dummy de utilidad.
4. **Posible redundancia con tc_sell_inspect.** El costo de venta más alto en años de
   inspección era la forma de Gillingham de capturar lo mismo sin un margen de
   reparación. Con A habría que decidir si se mantiene.

### Implicaciones para la identificación (Hu & Xin)

- **Si s se mide en la inspección** (lo natural: la inspección es donde se ve el
  estado), también se observa el s' de los coches que reprueban. Entonces la mezcla
  f(s' | h) = sum_r p(r|h) F_r(s'|h) **no cambia**: reprobar ocurre después de que se
  realiza s' y no lo sesga.
- **Si los reprobados desaparecen sin que se registre su s'**, la densidad observada se
  inclina por un factor conocido:

      f_obs(s' | h) ∝ (1 - kappa(s')) * sum_r p(r|h) F_r(s'|h)

  Ese factor es el mismo para ambos componentes de la mezcla. Si se conoce kappa, se
  divide y se recupera la mezcla de Hu & Xin. kappa se identifica comparando salidas
  en años con y sin inspección, dado s.
- **No añade decisiones nuevas.** La única elección no observada sigue siendo r, y la
  variable excluida sigue siendo R(j, a).

### Para la narrativa

"En mi modelo nadie decide sacar un coche del mercado. Las salidas son mecánicas
(accidentes y reprobar la inspección), pero su frecuencia depende de cuánto se reparó
antes. El zig-zag que Gillingham explica con chatarreo endógeno aquí sale de la
reparación preventiva."

---

## Idea B: reprobar abre una decisión de reparar o dar de baja

Si el coche reprueba, el dueño elige entre:

- **reparar para pasar:** paga R(j, a) y el coche sigue (con r = 1 ese periodo), o
- **no reparar:** el coche se da de baja, el dueño cobra p_scrap y queda como dueño de
  un terminal (puede comprar o quedarse sin coche).

La única decisión endógena sigue siendo reparar; la salida es la consecuencia de no
hacerlo. Es exactamente la historia de Gillingham, pero con el margen modelado.

### Cambios al modelo

1. **Una etapa nueva al inicio del periodo** para los estados activos con a en I:
   después de que se realiza kappa(s), los que reprobaron eligen
   {reparar para pasar, dar de baja} con un logit de escala sigma_insp. El valor
   esperado de entrar al periodo en x = (j, a, s), a en I, sería

       EV_inicio(x) = (1 - kappa(s)) * V(x) + kappa(s) * emax{ V_rep(x), V_baja(x) }

   - V(x) es el valor actual: keep / purge / trade y luego la etapa 2.
   - V_rep(x) es V(x) restringido a r = 1 y pagando R; hay que decidir si quien
     repara para pasar también puede vender después.
   - V_baja(x) = mu * p_scrap + (valor de un dueño terminal).
2. **El layout X no tiene que crecer.** La etapa de inspección se integra dentro de
   EV(x), igual que la etapa 2 se integra en W(h). Pero ahora la reparación es
   **forzada** (r = 1) para los que reprueban y la reparan, y eso cambia las filas de
   Ω/Q de esos estados. Hay que separar Pr(r = 1 | h) en "voluntaria" y "por
   inspección".
3. **El jacobiano βM sigue valiendo**, porque todo sigue siendo GEV encadenado, pero M
   gana la rama "reprobó y se dio de baja" (de X a terminal, dentro del mismo periodo).
   Conviene verificarlo de nuevo con `jax.jacfwd`.

### Implicaciones económicas

1. **Zig-zag en las salidas**, ahora endógeno vía la reparación:

       Pr(salir en inspección | x) = kappa(s) * (1 - pi(x))

   donde pi(x) = Pr(reparar para pasar | reprobó, x).
2. **Más respuesta a política.** Un subsidio a reparaciones (bajar R) reduce el
   scrappage en años de inspección. Con A solo lo reduce indirectamente, vía la
   reparación preventiva.
3. **Piso de precios.** Dar de baja garantiza al menos p_scrap a quien reprueba, lo que
   pone un piso a P en los estados malos. Ver el aviso de precios más abajo.

### Implicaciones para la identificación (Hu & Xin)

1. **Un momento nuevo con datos que sí existen.** El scrappage se observa en Gillingham,
   y en años de inspección es igual a acc(s) + kappa(s)(1 - pi(x)). Es una segunda
   ecuación sobre la reparación, independiente de la mezcla de s', así que sirve para
   sobreidentificar o para contrastar el estimador de Hu & Xin.
2. **Problema: kappa y pi no se separan solo con las salidas**, porque solo se ve su
   producto. Se separan con la variable excluida: R(j, a) mueve pi pero no kappa
   (Assumption 7 otra vez), y kappa es la misma en todos los años.
3. **Selección en la mezcla de los años de inspección.** Entre los coches que siguen
   después de la inspección, los que reprobaron y repararon tienen r = 1 con certeza.
   El peso de reparar en la mezcla de ese periodo ya no es p(r|h) sino

       [ (1 - kappa) * p_rep(h) + kappa * pi ] / [ (1 - kappa) + kappa * pi ]

   Esto pasa en las transiciones desde edades pares (4 -> 5, 6 -> terminal). Es una
   corrección conocida dado el modelo, pero el estimador tiene que incorporarla.
   Con A no aparece.
4. **Más parámetros a calibrar:** kappa(s) (k0, k1) y sigma_insp, además de decidir si
   la reparación para pasar es la misma r (mismo R, mismo efecto en s') o una distinta.
   Lo parsimonioso es que sea la misma: así R mueve las dos decisiones.

### Para la narrativa

"El chatarreo endógeno que Gillingham necesita para el zig-zag es en realidad una
decisión de no reparar. Al modelarla como tal, el scrappage observado en años de
inspección se vuelve información sobre la reparación no observada."

Ojo: aquí sí hay coches que salen por una decisión, aunque esa decisión sea "no
reparar". Si lo que incomoda es cualquier salida endógena, A es la opción.

---

## Comparación

| | A: reprobar = salida | B: reprobar -> reparar o dar de baja |
|---|---|---|
| Salidas por decisión | ninguna | sí, vía "no reparar" |
| Zig-zag en scrappage | sí | sí |
| Zig-zag en CCPs de reparar | sí (anticipación) | sí (anticipación + forzada) |
| Cambios al código | solo Q y `continuation_values` | etapa nueva en EV, Ω/Q con r forzada |
| Hu & Xin | sin cambios (o una reponderación conocida) | corrección de pesos en años de inspección |
| Momento extra con datos de Gillingham | kappa (no habla de r) | kappa(1 - pi): sí habla de la reparación |
| Parámetros nuevos | k0, k1 | k0, k1, sigma_insp |
| Piso para P | no | sí (p_scrap) |

## Aviso común: precios sin piso

Sin chatarreo endógeno nada impide que P(j, a, s) sea negativo para coches viejos y
frágiles: el vendedor le pagaría al comprador para deshacerse del coche. En Gillingham
la opción de chatarrear pone un piso cercano a p_scrap. Hay que revisarlo al resolver
ED(P) = 0:

- **Con A:** no hay piso. Si salen precios negativos, o se acepta (es una predicción del
  modelo), o se reintroduce un piso exógeno.
- **Con B:** el piso existe, pero solo para los que reprueban, en años de inspección.

## Dónde entra esto: modelo vs. estimación

Las dos ideas son parte del **modelo**, no solo de la estimación: cambian Q, por lo
tanto EV, las CCPs, la distribución estacionaria y los precios de equilibrio. Los
agentes anticipan la inspección. En el Monte Carlo hay dos piezas:

1. **Simulador** (el modelo con equilibrio). Si quieres datos con zig-zag, la inspección
   va aquí.
2. **Estimador** (Hu & Xin sobre los datos simulados).
   - Con A sin cambios, si s se mide en la inspección.
   - Con B hay que corregir los pesos de la mezcla en años de inspección.

Hay un caso en que solo afecta la estimación: simular con inspecciones y estimar
ignorándolas, para medir el sesgo de especificar mal el modelo. Es un ejercicio de
robustez, no el caso base.

Nada de esto viene de Gillingham: su modelo no tiene s, ni kappa, ni reparación. De ellos
solo se toma el hecho (zig-zag, inspección bianual desde los 4 años) y la
interpretación.
