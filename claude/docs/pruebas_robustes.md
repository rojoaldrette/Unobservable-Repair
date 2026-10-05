# Pruebas de robustez de la especificación

Cada prueba: **qué se cambia**, **qué pregunta responde** y **cómo** (simular con la
variante y estimar con el modelo base, o resolver el equilibrio con la variante y
comparar). Las marcadas con ★ son las que un evaluador probablemente pediría primero.

## 1. Transición de s (supuesto clave de Hu & Xin)

- ★ **Opción 1 contra opción 2.** Simular con la opción 1 (reparar baja ya la prob. de
  descomponerse) y estimar con la 2. ¿Cuánto sesgo genera el timing mal especificado?
  (`discusion_repair_o1.md`)
- ★ **eta heterocedástico:** s_sigma que depende de s o de la edad. Viola la Assumption 2
  (eta independiente de s). ¿Qué tan frágil es el estimador?
- **eta no normal** (t de Student, asimétrica). Hu & Xin no necesitan normalidad, pero
  el estimador paramétrico de la fase 4 sí.
- **Efecto de reparar heterogéneo:** s_repair por edad o marca, o multiplicativo
  (s' = (1 - delta r) m) en lugar de un desplazamiento aditivo.
- **Reparar como "reset":** s' vuelve cerca de s_new. Es un efecto grande y no lineal.
- **s_persist < 1:** el estado revierte a la media y la reparación pesa menos en el
  futuro.
- ★ **Frontera del grid:** E[eta | s] != 0 cerca de s_min y s_max por el recorte.
  Comparar con un grid más amplio, con renormalización o con reflexión, y medir el
  sesgo en esas celdas.
- **Tamaño del grid:** n_s = 50, 100, 200. ¿Cambian P, q y las CCPs?

## 2. Estructura de decisión

- ★ **Secuencial contra simultánea:** nested logit con sigma_repair <= sigma, o la
  versión anterior con repair al nivel de keep (solo los keepers reparan).
- ★ **sigma_repair** en {0.3, 0.5, 1, 2}. Cuánto ruido tiene la decisión no observada:
  con más ruido, las CCPs se acercan a 0.5 y la mezcla tiene menos información.
- **Quién repara:** el vendedor antes de vender en lugar del comprador después (cambia
  qué precio es P).
- **Coches nuevos reparables:** agregar R(j, 0).
- **Prohibir reparar en a_max - 1,** donde no tiene efecto en la transición.
- **Reparación con dos intensidades** (r en {0, 1, 2}): una mezcla de tres componentes.
  Hu & Xin lo permiten (rank, Assumption 4).

## 3. Variable excluida R(j, a)

- ★ **Cuánta variación:** número de "años" distintos y amplitud de R_t. Con poca
  variación la identificación es débil: graficar RMSE contra la dispersión de R.
- ★ **Violar la exclusión:** que R esté correlacionado con la calidad de la reparación
  (s_repair_t sube con R_t) o con la utilidad. ¿Qué sesgo aparece?
- **R común entre marcas** contra R por marca. ¿Basta con variación temporal?
- **Equilibrio por año:** si los años son equilibrios estacionarios distintos, ¿qué
  pasa si en realidad R cambia y el mercado no alcanza a ajustarse (transición)?

## 4. Heterogeneidad

- ★ **Varios tipos de hogar** (los 8 de Gillingham, distinto mu y u). Simular con
  heterogeneidad y estimar sin ella. Y la versión correcta: Hu & Xin con
  heterogeneidad no observada (sec. 6.1).
- **mu heterogéneo:** los pobres reparan más (o menos). Interactúa con R.

## 5. Mercado y equilibrio

- ★ **Unicidad:** resolver ED(P) = 0 desde varios P iniciales.
- **Piso de precios:** sin piso, con complementariedad y libre disposición (P >= p_scrap)
  o con chatarreo vía inspección (idea B). ¿Cambian las CCPs de reparar?
- **a_max mayor** (Gillingham usa 25) y J = 4 (agregar heavy green). ¿Los resultados
  dependen de lo corto de la vida del coche?
- **Costos de transacción:**
  - lectura de la Tabla 5: 0.9106 / 2.1929 (raw, la actual) contra 0.3454 / 0.9106 (v1.0),
    y 2.1929 como nivel o como incremento sobre 0.9106;
  - costo "no car" para dueños de terminal o no;
  - Tb = 0 y Ts = 0 (fricciones bajas, como el ejercicio de Gillingham en la sec. 4).
- **beta** en {0.90, 0.95, 0.98}. Afecta cuánto vale la reparación preventiva.
- **u_s** (desutilidad de manejar un coche frágil). Si es 0, reparar solo vale por la
  supervivencia.

## 6. Inspecciones (fase 5)

- Sin inspección, idea A, idea B.
- ★ **Estimar ignorando la inspección** cuando los datos sí la tienen.
- **Forma de kappa(s):** escalón contra logística, y su pendiente.
- **Edad de inicio y frecuencia** (bianual contra anual).

## 7. Medición y datos

- ★ **Solo datos agregados tipo daneses** (experimento puente, `fases_tesis.md` fase 6).
  Simular con s coche por coche, agregar a tasas de scrap por (j, a, t) en edades sin
  inspección, más CCPs y holdings por celda, y estimar con s latente. ¿Cuánto se pierde
  frente a Hu & Xin con s observado? Variantes: número de años, dispersión de R_t,
  s_sigma fijo o estimado, chatarreo voluntario en edades impares.
- ★ **s con error de medición:** s_obs = s + nu. Hu & Xin suponen s observado sin error;
  ¿cuánto se degrada el estimador?
- **s observado cada dos años** (como el odómetro danés): solo se ven (s_t, s_{t+2}),
  una mezcla de cuatro componentes.
- **Tamaño de muestra:** N y T. Curva de RMSE contra N.
- **Salidas no registradas:** perder el s' de los coches que se descomponen o reprueban
  (selección).

## 8. Numéricas

- Tolerancias (sa_tol, vfi_tol) y float32 contra float64.
- Solo successive approximations contra SA + Newton-Kantorovich: misma EV.
- Jacobiano analítico contra autodiff (ya está en las pruebas del script).

## Prioridad sugerida

1. sigma_repair y cuánta variación hay en R (definen si hay algo que identificar).
2. Opción 1 contra opción 2.
3. eta heterocedástico y frontera del grid (supuesto clave de Hu & Xin).
4. Heterogeneidad de hogares.
5. s con error de medición.
6. Ignorar la inspección.
7. Solo datos agregados tipo daneses (puente hacia la fase 6).
