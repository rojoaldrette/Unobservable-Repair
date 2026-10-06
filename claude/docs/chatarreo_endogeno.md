# Chatarreo endógeno en modelo_fin: como opción

Estado: **implementado en v1.8** (`Params.scrap`, `sigma_sell`).  Apagado en `Params()`
(el modelo de antes); **encendido en la calibración "tesis"**, porque sin él los precios
de coches viejos salen negativos con a_max = 25 (ver `calibracion.md`).  Pruebas: con
`scrap = False` todo da lo mismo que antes; gradiente con chatarreo contra diferencias
finitas 1e-9.

## Por qué se quitó (v1.0)

Para que lo observado no viniera de tres fuentes mezcladas (reparación, chatarreo y
accidentes) y la identificación de la reparación fuera limpia.

## Por qué se puede regresar sin romper la identificación

### 1. El orden en el tiempo protege la mezcla de Hu & Xin

```
inicio t:  x = (j, a, ℓ)  -> etapa 1: keep / vender / chatarrear / trade -> h
           -> etapa 2: reparar o no (no se observa) -> manejas
           -> accidente con prob s(h)
inicio t+1: x' = (j, a+1, ℓ')     <- aquí se mide ℓ'
           -> etapa 1 de t+1: aquí se decide chatarrear, ya con ℓ' observado
```

La reparación deja huella en ℓ -> ℓ'. El chatarreo es una decisión de la etapa 1 del
periodo siguiente, tomada sobre ℓ' ya medido. Por eso

    f(ℓ' | h) = (1 − p(h)) F_0(ℓ' | h) + p(h) F_1(ℓ' | h)

es la misma con o sin chatarreo. El chatarreo es una CCP más (como keep o trade), que
depende del estado; no entra en la transición. El simulador ya guarda `s_next` al
inicio del periodo siguiente, antes de la elección (`gen_dataset.py:161`).

### 2. Las tres salidas se separan porque s se observa

En Gillingham, accidente y chatarreo se confunden: la prob. de accidente es un logit en
(j, a) con parámetros por estimar y se separan solo por forma funcional.

Aquí **s es la probabilidad de accidente y se observa**: Pr(accidente | h) = s(ℓ_h) =
logística(ℓ_h), sin parámetros. Con la opción 2 no depende de r.

| salida | depende de | cómo se ve en los datos |
|---|---|---|
| accidente | s(ℓ_h), conocido | activo en t, terminal en t+1 |
| chatarreo | CCP en x' = (j, a+1, ℓ') | activo en t+1, sale sin comprador |
| edad terminal | a = a_max | determinista |
| reparación | solo la media de ℓ' | mezcla sobre ℓ' (Hu & Xin) |

### 3. Qué sí lo rompería

- **ℓ' medido solo en coches que no se chatarrean** (p. ej. si s viene de una inspección
  a la que solo llegan los que siguen). La muestra de ℓ' queda truncada por una decisión
  que depende de ℓ': la primera etapa de Hu & Xin tendría sesgo de selección. La
  verosimilitud estructural lo corrige multiplicando por Pr(no chatarrear | x'), que el
  modelo da, pero la primera etapa sola no.
- **Opción 1 de reparar** (reparar baja el accidente de este periodo): las salidas
  dependen de r y las tres fuentes sí se enredan. Otro argumento para la opción 2.
- **La fase 6 de `fases_tesis.md`** (datos daneses agregados, sin s) usa "scrap =
  accidente en edades impares". Con chatarreo encendido ese supuesto es falso; ahí hay
  que dejarlo apagado o tratarlo como la variante "un poco de chatarreo voluntario" que
  ya está listada.

## Por qué conviene tenerlo

1. **Anidar a Gillingham.** Sin chatarreo, el modelo sin reparación no reproduce G25 (el
   42% de sus salidas es voluntario). Con él, la comparación "mismos parámetros, con y
   sin reparar" cambia una sola cosa.
2. **Es el margen económico relevante.** Para un coche viejo la alternativa a reparar es
   chatarrearlo, no vendérselo a otro hogar.
3. **Piso de precios (pendiente 6 de v1.0).** Nadie vende por menos de lo que le da la
   chatarra, así que los precios negativos de coches malos desaparecen.
4. **El ejercicio del sesgo.** Datos con reparación estimados con Gillingham: parte de
   "reparar en vez de chatarrear" se lee como preferencias o costos. El proceso que
   genera los datos tiene que traer los dos márgenes.

## Especificación (como Gillingham, ec. 18)

Al deshacerse de un coche activo (en purge o en trade), nido vender / chatarrear con
escala sigma_sell (0 < sigma_sell <= sigma_trade):

    vender:      mu P(j, a, ℓ) − Ts(a)
    chatarrear:  mu p_scrap_j
    disposal(x) = sigma_sell · log( exp((mu P − Ts)/sigma_sell) + exp(mu p_scrap/sigma_sell) )
    Pr(scrap | x, deshacerse) = logística( (mu p_scrap − mu P + Ts) / sigma_sell )

- Es la misma dentro de purge y de trade (decisión estática), como en el paper.
- Coche terminal: solo chatarra, como hoy.
- Oferta: S(j, a, ℓ) = q(j, a, ℓ) (1 − keep) (1 − scrap). Lo chatarreado sale del mercado.
- Parámetro nuevo: sigma_sell = 0.3454 (Tabla 5, lectura raw).
- `scrap = False`: disposal = mu P − Ts, exactamente el modelo actual.

## Pruebas que garantizan "no rompe nada"

1. `scrap = False` da lo mismo que el código actual, a precisión de máquina (EV, CCPs, q,
   equilibrio).
2. **Anidamiento:** con `scrap = True`, sin reparación (Pr(repair) = 0), s_sigma -> 0 y
   u_s = 0, ℓ es determinista y el modelo es el de Gillingham. Su equilibrio debe
   coincidir con el de `gillingham/` (a_max chico, en local).
3. **La mezcla no cambia:** en datos simulados con chatarreo, la primera etapa de Hu &
   Xin (`loglikelihood.py`) recupera s_repair y p_t(h) igual que sin chatarreo (prueba de
   humo chica en local; el MC completo va a GPU).
4. Filas de Ω y Q suman 1, dT/dEV == βM, q = qM (las de siempre).

## En la verosimilitud estructural

Se agrega o ∈ {keep, purge vende, purge chatarrea, trade vende, trade chatarrea}, como en
la réplica de Gillingham. Dos versiones de datos para el MC:

- **ideal:** accidente y chatarreo se distinguen (la ruta del coche se ve);
- **tipo daneses:** solo "salida". Por la sección 2 debería costar poca precisión porque
  s(ℓ_h) da la tasa de accidentes.

Ver `verosimilitud_estructural.md`.
