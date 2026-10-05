# Estructura de la decisión: ¿repair al nivel de keep o después?

## Tu lectura del código anterior era correcta (con un matiz)

En una versión anterior del código la etapa superior era un solo logit sobre

    {keep, repair, purge, trade}

o sea, reparar era una alternativa hermana de keep. Dos consecuencias:

1. **Solo quien se quedaba el coche podía reparar.** Quien compraba un usado h nunca lo
   reparaba ese periodo, aunque estuviera en el mismo estado (j, d, s) que un keeper.
2. **Todos los shocks llegaban a la vez.** El individuo veía el shock de reparar antes de
   decidir si se quedaba el coche.

El matiz: con sigmas iguales es un logit multinomial y, por IIA,

    Pr(repair | keep o repair) = 1 / (1 + exp(-(v_repair - v_keep)/sigma))

ya no dependía de los valores de trade o purge. La "independencia" que buscas existía
para los keepers. Lo que faltaba era que valiera para cualquier tenencia h y que el
timing fuera secuencial.

## Estructura implementada ahora

Dentro del periodo, para un dueño de un coche activo x = (j, a, s):

    Etapa 1 (shocks eps_1):  keep  |  purge (vender, quedarse sin coche)  |  trade (vender, comprar h)
                                     └── dentro de trade: qué h comprar (nido sigma_trade)
    -> ahora tienes h (el mismo coche si keep, el comprado si trade)
    Etapa 2 (shocks eps_2, nuevos e independientes):  reparar o no reparar h
    -> manejas h, se descompone con prob s o transita s' ~ F_r

Valor de la etapa 2 para un usado h:

    v_0(h) = beta * E[EV(x') | h, r=0]
    v_1(h) = -mu * R(j, d) + beta * E[EV(x') | h, r=1]
    W(h)   = sigma_repair * log( exp(v_0/sigma_repair) + exp(v_1/sigma_repair) )
    Pr(repair | h) = exp(v_1/sigma_repair) / (exp(v_0/sigma_repair) + exp(v_1/sigma_repair))

La etapa 1 ve a h solo a través de W(h):

    keep(x)  = u(x) + W(x)
    buy(h)   = u(h) - mu*P(h) - Tb + W(h)          (usados)
    buy(new) = u(new) - mu*p_new - Tb + beta E[EV | new]   (los nuevos no se reparan)

`Pr(repair | h)` depende solo de h: es la misma si llegaste a h quedándote el coche o
comprándolo. Eso es lo que pedías.

## La analogía con el scrap endógeno de Gillingham: vale a medias

Gillingham (sec. 3.3, ec. 18) muestra que, con costos de transacción aditivamente
separables y el mismo sigma_s en todos los nidos, la decisión de vender o chatarrear es
**estática**: su probabilidad depende solo de (i, a) y **no afecta el futuro**. Por eso se
factoriza como pi(j, d, s | i, a) = pi(j, d | i, a) * pi(s | i, a).

Reparar comparte la primera propiedad pero no la segunda:

| | scrap en Gillingham | repair aquí |
|---|---|---|
| Prob. independiente de la decisión anterior | sí, depende solo de (i, a) | sí, depende solo de h |
| Afecta el futuro | no, es estática | **sí**: cambia la distribución de s' |
| La etapa anterior depende de ella | solo vía una constante (el valor inclusivo de vender/chatarrear) | **sí**, vía W(h), que depende de EV |

Entonces "la prob. de repair es independiente de la decisión anterior" es correcto, pero
la relación al revés no se cumple: keep/trade sí depende de reparar, porque un coche que
se puede reparar vale más (W(h) >= v_0(h)). Esto tiene que ser así: si reparar no moviera
la decisión de quedarse o vender, tampoco movería los precios P y el Monte Carlo no
tendría nada que ver con el equilibrio.

## ¿Secuencial o nested logit simultáneo?

Con errores EV1 las dos interpretaciones dan exactamente las mismas fórmulas: W(h) es el
valor inclusivo de un nido {r=0, r=1} debajo de cada h. Solo cambia la restricción sobre
los parámetros:

- **Nested logit simultáneo**: requiere sigma_repair <= sigma_trade <= sigma (McFadden).
- **Secuencial** (eps_2 se realiza después de la etapa 1): no hace falta ninguna
  restricción; basta sigma_repair > 0.

En el código uso la lectura secuencial y solo pido `sigma_repair > 0`. Con los defaults
(todos los sigma = 1) las dos coinciden.

## Lo que esto le da a Hu & Xin

En la notación de la sección 6.3 (ec. 6.4): a1 = (keep/purge/trade, qué h compró)
se observa y a2 = r no se observa. Con la estructura secuencial,

    p(a2 | x, a1) = Pr(repair | h)

así que la transición observada de s para un usado h,

    f(s' | h, sobrevive) = (1 - p_rep(h)) F_0(s' | h) + p_rep(h) F_1(s' | h),

es la misma mezcla para keepers y para quien acaba de comprar h. En el Monte Carlo se
pueden juntar las dos poblaciones: más observaciones por cada h. Esto lo hace
`observed_s_transition`.

## Detalles a decidir

1. **Coches nuevos no se reparan.** Lo mantuve como antes. Si quieres permitirlo, haría
   falta una columna R(j, 0).
2. **Reparar en la última edad activa (a = a_max - 1) no cambia la transición:** el coche
   llega a terminal con certeza, así que v_1 = v_0 - mu*R. Aun así,
   Pr(repair) = 1 / (1 + exp(mu R / sigma_repair)) > 0. No es un error: en el logit cada
   alternativa trae su shock eps, que representa un beneficio idiosincrático de reparar
   no modelado, y quien saca eps_1 - eps_0 > mu R repara. Con R x5 sale 0.05 (antes
   0.36). Si se quiere exactamente 0, hay que quitar la opción en esa edad. Para Hu & Xin
   da igual, porque ahí no hay s' que observar.
3. **Quién repara antes de una venta.** Gillingham interpreta Ts como el costo de
   "undertaking repairs and improvements to make the car acceptable to potential buyers".
   Aquí el comprador repara después de comprar y P(h) es el precio del coche **sin**
   reparar. Si en tus datos el vendedor es quien repara antes de vender, el timing
   cambiaría (la reparación iría en la etapa 1 del vendedor y P sería de un coche ya
   reparado).
