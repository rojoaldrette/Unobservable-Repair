# Idea: el kilometraje como el estado observado de Hu & Xin

Estado: **propuesta, no implementada.**

## El problema que resuelve

El teorema de Hu & Xin necesita un estado continuo observado **por coche** en periodos
consecutivos, cuya transición mueva la reparación (`id_modelo_vsHu.md`). En los datos
daneses s no existe por coche, pero **el odómetro sí**: Gillingham lo usa para su
regresión de manejo con 19.6 millones de periodos de manejo (Tabla 6), medidos en cada
inspección.

## La idea

**Un coche en peor estado se maneja menos.** Se descompone, está en el taller, el dueño
no se fía de él para viajes largos. Entonces el kilometraje entre inspecciones es una
señal del estado del coche, y reparar, al mejorar el estado, **sube el kilometraje
futuro**.

En vez de pedirle a los datos s, se usa como estado de Hu & Xin la **intensidad de uso**:

    k_t = km por año del coche en el ciclo de inspección t
          (diferencia de odómetro entre inspecciones / años entre ellas)

y la transición

    k_{t+1} = m_k(r_t, k_t, j, a, tipo de hogar) + e_{t+1},   E[e | k] = 0, e indep. de r

Reparar desplaza la media del kilometraje futuro: es la Assumption 2 de Hu & Xin, con k
en lugar de s.

## Por qué encaja con los datos daneses

1. **Por coche y en ciclos consecutivos:** cada inspección da una lectura del odómetro
   del mismo coche.
2. **El ciclo de inspección como periodo de la reparación.** Las inspecciones son
   bianuales desde los 4 años y las reparaciones se concentran en la inspección (si el
   coche reprueba, hay que reparar). Es natural suponer **una decisión de reparar por
   ciclo**, tomada en la inspección. La transición k_t -> k_{t+1} es entonces una
   mezcla de **dos** componentes, no de cuatro, y el periodo del estimador es el ciclo
   de 2 años.
3. **Medido antes de reparar:** la lectura en la inspección t refleja el uso del ciclo
   anterior, antes de la reparación de esa inspección. El timing es el que pide la
   opción 2: reparar mueve el futuro, no el pasado.
4. **Variable excluida:** R_t (por ejemplo, el IPC danés de reparación, COICOP 07.2.3)
   cambia la probabilidad de reparar, pero no cuánto se maneja un coche **dado** su
   estado y su reparación.
5. **Elección observada a1:** keep/trade entre inspecciones. Conviene usar solo coches
   **que no cambiaron de dueño** dentro del ciclo, para que el cambio de k no venga de
   que lo maneja otro hogar. Eso es condicionar en a1 = keep, como en la ec. 6.4.

## Cambios al modelo

Siguiendo a Gillingham (sec. 6.1, ecs. 46–47), el manejo es una decisión estática dentro
del periodo:

    x*(j, a, s, tipo) = (1/phi) * [ -mu p_j + gamma_0 + gamma_1 a - gamma_s s + omega ]

- gamma_s > 0: peor estado, menos manejo. Este término es **nuevo**; Gillingham supone
  que el deterioro no depende del manejo ni al revés.
- Al sustituir x* en la utilidad sale la u(j, a, s) indirecta: cuadrática en a y ahora
  también en s. El término u_s del código actual pasa a tener un fundamento: es la
  pérdida de utilidad de manejar menos.
- El simulador del Monte Carlo genera, además de s, el kilometraje de cada coche y ciclo.

Dos formas de usarlo en la estimación:

- **Versión simple (k como estado).** Se supone que el estado relevante está resumido en
  k: s se vuelve una función de k, o directamente se modela el estado como "intensidad
  de uso". Hu & Xin se aplica tal cual sobre k.
- **Versión con medición (k mide s con error).** k_t = f(s_t) + omega_t, con s latente.
  k ya no es Markov por sí solo y hace falta el marco de error de medición (Hu 2008;
  Hu & Shum 2012; Apéndice B de Hu & Xin con una segunda medida, por ejemplo
  sobrevivir). Es más realista y más difícil.

## Riesgos y cómo atacarlos

| riesgo | por qué importa | qué hacer |
|---|---|---|
| El manejo depende del hogar (distancia al trabajo, ingreso, precio de gasolina) más que del coche | la mezcla en k reflejaría al hogar, no a la reparación | condicionar en el tipo de hogar (8 tipos de Gillingham) y en el precio de combustible del ciclo; usar solo coches con el mismo dueño |
| Choques de manejo persistentes (cambio de trabajo) | violan "e independiente de r" | controles del hogar; probarlo en el Monte Carlo como prueba de robustez |
| R podría afectar el manejo por otra vía (ingreso disponible) | viola la exclusión | con mu cuasi-lineal el efecto ingreso es nulo en el modelo; discutirlo |
| La reparación puede no mover el manejo (gamma_s chico) | m_k(1, k) ≈ m_k(0, k): la Assumption 3 falla y no se identifica | es contrastable con los datos (sec. 6.3 de Hu & Xin): si E[k_{t+1} \| k_t] no cambia con R_t, no hay señal |
| Ciclos solo desde los 4 años | no hay información de reparación en coches jóvenes | aceptarlo: la reparación importa sobre todo en coches viejos |

## Experimento de Monte Carlo propuesto

1. Agregar manejo al modelo con gamma_s > 0 y simular k por coche y ciclo.
2. Estimar con Hu & Xin sobre k (versión simple) usando solo coches sin cambio de dueño.
3. Comparar con Hu & Xin sobre s (fase 4) y con el modelo de celdas (experimento
   puente). Así quedan tres escenarios de datos: ideal (s), realista (km) y agregado
   (celdas).
4. Variar gamma_s y el ruido del manejo: **¿qué tan fuerte tiene que ser la relación
   entre estado y uso para identificar?**

## Otra fuente con la misma lógica: la verificación vehicular en México

La verificación de emisiones en CDMX y su zona metropolitana es **semestral**, mide por
coche **variables continuas** (emisiones de gases, y en algunos centros el odómetro) y
ocurre **antes** de reparar: si el coche no pasa, se repara y se vuelve a verificar. En
México buena parte de la reparación es **informal**, así que no deja registro: es
exactamente una decisión no observada. Las emisiones del coche son un estado continuo
(miden el deterioro del motor) y reparar las baja.

Por verificar:
- si SEDEMA, el portal de datos abiertos de CDMX o la PROFECO publican resultados por
  vehículo (aunque sea anonimizados);
- si hay identificador del coche entre verificaciones;
- qué variables vienen.

Si existe, sería una aplicación con el teorema completo de Hu & Xin y un contexto (la
reparación informal) que hace la pregunta de la tesis muy relevante.
