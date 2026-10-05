# Réplica de Gillingham et al. (`claude/scripts/gillingham/`)

Modelo de la sec. 3 de Gillingham, Iskhakov, Munk-Nielsen, Rust & Schjerning (2019 WP),
con los parámetros estimados de la sec. 6 para un solo tipo de hogar ("Low WD, Couple,
Poor") y tres marcas (light brown, light green, heavy brown). Sirve para comparar su
equilibrio con el de `modelo_fin`.

## Archivos

| archivo | qué tiene |
|---|---|
| `params.py` | `GParams` con fuentes de cada número |
| `utils.py` | layout X = [act(j,a) \| term(j) \| none], H = [used(j,d) \| new(j) \| none] |
| `bellman.py` | utilidad, accidentes (logit, Tabla 4), Ts, valores, `T`, `solve_bellman` |
| `probabilities.py` | CCPs, incluida la de chatarrear (ec. 18) |
| `transitions.py` | Q, M = ΩQ, distribución estacionaria |
| `ED.py` | ED_log = log D − log S y Newton con jacobiano denso (72 incógnitas) |
| `tests.py` | `python tests.py [--a_max 7]` (incluye solver conjunto y gradiente implícito) |
| `../comparacion/compare.py` | corre G25, G7, M7 y M7_gill y escribe `comparacion/output/*.csv` |
| `theta.py` | `GModel(g, th)`: parámetros estructurales como pytree dinámico; vector libre x <-> th |
| `equilibrium.py` | equilibrio rápido: Newton sobre z = (EV, P) conjunto, jiteado con th dinámico |
| `simulate.py` | panel de hogares desde q y agregación a celdas (ec. 40) |
| `loglikelihood.py` | verosimilitud DNFXP parcial (apéndice D) y completa; gradiente implícito; L-BFGS + BHHH |
| `montecarlo.py`, `main.py` | Monte Carlo del estimador (ver "Estimación y Monte Carlo") |

Todos los `*_raw` (`T_raw`, `ccps_raw`, `physical_matrix_raw`) son las versiones sin jit:
aceptan un `GParams` o un `GModel`.  `T`, `ccps` y `physical_matrix` siguen siendo los
mismos (jit con g estático).

## Modelo

- **Estado:** (j, a), a = 1..a_max, más "sin coche". No hay s; el accidente es
  alpha(j, a) = logit(int_j + age_j a) (Tabla 4).
- **Elecciones** (figura 1): purge | keep | trade a (j, d). Al deshacerse del coche:
  vender (cobra mu P − Ts) o chatarrear (cobra mu p_scrap), en un nido con escala
  sigma_sell. Por la ec. 18 es una decisión estática y es la misma dentro de purge y
  de trade.
- **Equilibrio:**
  - D(j,a) = trade_mass · Pr(j,a | trade);
  - S(j,a) = q(j,a)(1 − keep)(1 − scrap): lo chatarreado sale del mercado.
- **Árbol de compra:** los nidos de marca y edad del paper se colapsan (sigma_trade
  único). El paper no reporta escalas distintas de 1 salvo la de vender/chatarrear.

## Parámetros y su origen

| parámetro | valor | fuente |
|---|---|---|
| mu | 0.1131 | Tabla 7 |
| u0, u1 | (3.649, 3.113, 5.154), (−0.146, −0.092, −0.220) | Tablas 8, 9 |
| u2 | 0 | no se reporta |
| tc_buy, tc_buy_nocar | 6.5944, 1.7899 | Tabla 10 |
| sigma_sell | 0.3454 | Tabla 5 (lectura raw) |
| tc_sell | 0.9106 | Tabla 5 (lectura raw) |
| tc_sell_inspect | 2.1929 | Tabla 5 (lectura raw; estimación −2.1929 leída como coeficiente en la utilidad de vender) |
| accidentes | int (−5.62, −6.04, −5.67), edad (0.180, 0.222, 0.202) | Tabla 4 |
| p_new | (142.33, 110.09, 275.84) | Tabla 3 |
| p_scrap, beta | 2.0, 0.95 | no se reportan (mismos que modelo_fin) |
| u_even (dummy de edad par) | 0 | se menciona en la sec. 6.2, pero no está en las tablas |

### Corrección importante: la Tabla 5

Con `pdftotext -layout` las etiquetas de la Tabla 5 se desalinean. En modo `-raw` se
leen así:

    lambda_s                                   0.3454
    sales transaction cost                     0.9106
    sales transaction cost (inspection year)  -2.1929

- **0.3454 es la escala del nido vender/chatarrear**, no un costo de venta.
- **En `modelo_fin` v1.0 se usó otra lectura:** tc_sell = 0.3454 y tc_sell_inspect =
  0.9106. Probablemente está mal.
- **El −2.1929 se interpreta como coeficiente de la utilidad de vender en año de
  inspección**, es decir, un costo de 2.19. Así vender es más caro en edades pares,
  hay más chatarreo y aparece el zig-zag de la sección 6.2.
- **Por confirmar:** los errores estándar de la tabla son idénticos a las
  estimaciones, señal de un problema de formato. Conviene revisar la versión publicada
  (JPE) o el código de los autores.

## Comparación de equilibrios (`comparacion/compare.py`, n_s = 24 para modelo_fin)

| modelo | qué es | sin coche | masa que compra | nuevos / compras | tiempo |
|---|---|---|---|---|---|
| G25 | Gillingham, a_max = 25 (paper) | **0.016** | 0.089 | 0.63 | 21 s |
| G7 | Gillingham, a_max = 7 | **0.728** | 0.054 | 0.74 | 17 s |
| M7 | modelo_fin, defaults v1.0 | **0.916** | 0.025 | 0.55 | 134 s |
| M7_gill | modelo_fin, s calibrada a la Tabla 4 + Ts raw | **0.649** | 0.069 | 0.74 | 161 s |

Light brown (j = 0), edades 1..6. P: precio en miles de DKK. keep / trade / purge:
probabilidades por estado (ponderadas por q en s para modelo_fin). acc: Pr(accidente).
rep: Pr(repair).

| a | P G7 | P M7_gill | P M7 | keep G7 | keep M7_gill | keep M7 | purge G7 | purge M7 | acc G7 | acc M7 | rep M7_gill |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 137.5 | 139.0 | 131.1 | 0.894 | 0.894 | 0.743 | 0.079 | 0.225 | 0.004 | 0.042 | 0.171 |
| 2 | 112.9 | 113.6 | 109.9 | 0.893 | 0.892 | 0.732 | 0.079 | 0.235 | 0.005 | 0.043 | 0.133 |
| 3 | 88.3 | 88.6 | 86.7 | 0.893 | 0.890 | 0.724 | 0.079 | 0.241 | 0.006 | 0.058 | 0.104 |
| 4 | 69.6 | 69.7 | 66.7 | 0.942 | 0.940 | 0.776 | 0.043 | 0.196 | 0.007 | 0.082 | 0.081 |
| 5 | 39.6 | 39.7 | 42.1 | 0.892 | 0.888 | 0.703 | 0.080 | 0.260 | 0.009 | 0.113 | 0.063 |
| 6 | 22.3 | 21.0 | 22.1 | 0.917 | 0.938 | 0.753 | 0.062 | 0.216 | 0.011 | 0.151 | 0.049 |

### Lectura

1. **modelo_fin anida a Gillingham.** Con s calibrada a sus tasas de accidente
   (M7_gill), precios y CCPs quedan casi idénticos a G7: diferencias de 1–2 mil DKK en
   P y menores a 0.02 en keep. Reparar y el nido de s no rompen la estructura de
   Gillingham. Esto sirve como validación en la tesis.
2. **Menos hogares sin coche en M7_gill (0.65) que en G7 (0.73).** Poder reparar sube
   el valor de tener coche. Es el efecto de la reparación en el margen extensivo.
3. **a_max = 7 explica la mayor parte de los hogares sin coche:** 0.73 en G7 contra
   0.016 en G25, con los mismos parámetros. Los parámetros de Gillingham suponen coches
   que viven 25 años. Comparar con datos exige subir a_max.
4. **La calibración de s de v1.0 (M7) está fuera de rango:**
   - accidentes de 4–15% contra 0.4–1.1%;
   - purge de 0.20–0.26 contra 0.04–0.08;
   - 92% de hogares sin coche.
5. **El zig-zag por inspección aparece** en keep (sube en a = 4 y 6) en G7 y M7_gill,
   por el Ts de año de inspección. En G25 aparece también en precios y en el chatarreo
   endógeno de edades altas, como en la figura 7 del paper.

## Estimación y Monte Carlo (v1.3)

El paper **no reporta un Monte Carlo**: estima con el registro danés. Aquí se replica su
estimador (sec. 5.1 y apéndice D) y se aplica a datos simulados del propio modelo.

### Estimador (DNFXP, como en el paper)

- **Datos:** panel de N hogares y K años, con el estado inicial sacado de la distribución
  estacionaria q. Se agrega a celdas de transiciones h_{t-1} -> (x_t, o_t, h_t) con conteos
  (ec. 40). o en {keep, purge vende, purge chatarrea, trade vende, trade chatarrea}.
- **Verosimilitud "parcial" (la del paper):**
  - no se observan precios ni accidentes;
  - un coche que sale del parque puede ser accidente (x = terminal) o chatarreo endógeno
    (x activo y elige chatarrear): Pr(o, h' | h) = (Q C)[h, o] · buy(h')^{1(trade)}
    (ecs. 75-77);
  - P entra como P(θ), resuelto en cada evaluación.
- **Verosimilitud "completa" (oráculo):** ve x, incluido el accidente:
  Q[h, x] · C[x, o] · buy(h').
- **Equilibrio en cada evaluación:** Newton sobre z = (EV, P) conjunto, jiteado con θ
  dinámico (`GModel`). Tarda ~3 s en frío (contra 26 s del solver de `ED.py`, mismo
  resultado a 1e-10) y ~6 ms en caliente.
- **Gradiente analítico** por la función implícita, dz/dθ = −F_z⁻¹ F_θ, como en el paper.
  Contra diferencias finitas coincide a 5e-8. Da los scores por celda, de donde salen el
  gradiente y la matriz BHHH.
- **Optimización:** L-BFGS con el gradiente analítico y luego BHHH (el del paper) para
  terminar y sacar errores estándar.
  - BHHH solo, desde arranques lejanos, avanzaba a pasos minúsculos (el producto
    exterior de scores aproxima mal el hessiano lejos del óptimo).
  - BHHH se detiene si la LL no sube en 3 iteraciones seguidas.
- **18 parámetros:** mu, u0 (3), u1 (3), tc_buy, tc_buy_nocar, tc_sell, tc_sell_inspect,
  sigma_sell, acc_int (3), acc_age (3). Normalizaciones: sigma = 1, u_none = 0. Fijos
  (conocidos): p_new, p_scrap, beta.
- **Arranques:** la verdad más 2 perturbados (x0 + 0.1 |x0| N(0,1)); se queda el de mayor LL.

### Diseño

- a_max = 25 (el del paper), N = 20,000 hogares, K = 10 años (~180,000 transiciones),
  50 réplicas.
- Equilibrio verdadero: 1.6% sin coche. Por coche en circulación al año: 2.4% de
  chatarreo endógeno y 3.2% de accidentes, así que el 42% de las salidas es voluntario.
- Dos diseños:
  - **p18:** los 18 parámetros libres;
  - **p17:** tc_buy fijo en la verdad.

```
cd claude/scripts/gillingham
python main.py --reps 0:50 -v                 # p18
python main.py --reps 0:50 --fix tc_buy -v    # p17
python main.py --summarize [--fix tc_buy]
```

- **Salida:** `claude/output/montecarlo/gillingham/`, con los CSV por réplica y
  `resumen_*.csv` (sesgo, RMSE, se medio contra sd del MC, cobertura al 95%).
- **Tiempo:** ~30 s por réplica (2 estimadores x 3 arranques).

### Resultados (50 réplicas)

Selección. Las tablas completas están en `resumen_N20000_K10_A25_p18.csv` y `..._p17.csv`.

| | verdad | p18 parcial: sesgo / se / cob. | p18 completa | p17 parcial | p17 completa |
|---|---|---|---|---|---|
| mu | 0.1131 | 0.004 / 0.031 / 0.98 | 0.004 / 0.020 / 0.92 | 0.001 / 0.0056 / 0.94 | 0.000 / 0.0039 / 0.94 |
| tc_buy | 6.594 | −0.35 / 3.81 / 1.00 | −0.42 / 2.34 / 0.92 | fijo | fijo |
| tc_sell | 0.911 | 0.35 / 3.81 / 1.00 | 0.42 / 2.34 / 0.92 | 0.001 / 0.021 / 0.98 | 0.001 / 0.020 / 0.92 |
| sigma_sell | 0.345 | 0.005 / 0.029 / 0.96 | 0.000 / 0.021 / 0.94 | 0.006 / 0.029 / 0.92 | 0.000 / 0.021 / 0.94 |
| acc_int_0 | −5.625 | 0.009 / 0.159 / 0.94 | −0.009 / 0.108 / 0.96 | 0.011 / 0.159 / 0.94 | −0.009 / 0.108 / 0.96 |
| acc_age_0 | 0.180 | −0.003 / 0.019 / 0.94 | 0.001 / 0.008 / 0.96 | −0.003 / 0.019 / 0.96 | 0.001 / 0.008 / 0.96 |
| % salidas voluntarias (0.419) | | 0.425 (RMSE 0.021) | 0.419 (0.005) | 0.425 (0.022) | 0.419 (0.005) |
| RMSE de P (miles DKK) | | 17.9 | 9.9 | 0.93 | 0.55 |

### Lectura

1. **El estimador del paper funciona.** No tiene sesgo apreciable, los se por BHHH
   coinciden con la dispersión del MC y la cobertura queda en 0.88-1.00 (casi siempre
   0.92-0.96). Converge en el 100% de las réplicas.
2. **El chatarreo endógeno se separa de los accidentes sin observar accidentes.** Con la
   verosimilitud parcial, la fracción voluntaria de las salidas sale 0.425 (verdad
   0.419), y acc_int y acc_age se recuperan. Lo que lo identifica es la forma funcional:
   accidentes logit en la edad contra chatarreo por la comparación precio − Ts contra
   p_scrap, con el zig-zag de inspección. Ver los observados cuesta precisión (se de
   acc_age_0: 0.019 contra 0.008), no sesgo.
3. **Sin precios observados, tc_buy y tc_sell casi no se identifican por separado.**
   - Subir tc_buy en δ y bajar tc_sell (y tc_sell_inspect) en δ se compensa con P -> P − δ/mu en
     todas las transacciones de usados.
   - Solo lo rompen las compras de coches nuevos (p_new fijo) y el chatarreo (p_scrap
     fijo).
   - Resultado: se de 3.8 utils para tc_buy y tc_sell. Entre réplicas, tc_buy y tc_sell
     tienen correlación −1.000, y |corr| = 0.98 con mu. Eso arrastra a mu (se 0.031
     contra 0.0056) y a los precios (RMSE 18 mil DKK contra 0.9).
   - Ver los accidentes no lo arregla (RMSE de P de 9.9). Fijar tc_buy, u observar
     algunos precios, sí.
   - **Esto es un artefacto de tener un solo tipo de hogar** (ver "Tipos de hogar"
     abajo): con dos tipos de mu distinta, como en el paper, la dirección se identifica.
     Con un solo tipo hay que anclarla: fijar tc_buy, usar precios de usados, o meter
     heterogeneidad en mu.
4. **Máximos locales.**
   - En p17 parcial, 3 de 50 réplicas tuvieron un arranque perturbado que terminó en
     un punto peor; en p18, ninguna.
   - En una prueba previa con N = 50,000, un arranque perturbado se fue a una solución
     degenerada de la marca 2 (u0_2 = −38, la marca con 1.5% del parque), con menor LL.
   - Hay que usar varios arranques.
5. **tc_buy_nocar** sale igual con las dos verosimilitudes: solo lo informan los hogares
   sin coche, cuyo estado sí se observa.

### Pendientes

- Precios de usados como dato adicional (verosimilitud conjunta, o momentos) para
  identificar tc_buy y tc_sell por separado.
- Tipos de hogar (el paper tiene 8) y el dummy de edad par en la utilidad.
- Llevar la misma maquinaria (`GModel`, Newton conjunto, gradiente implícito) a
  `modelo_fin` para la segunda etapa estructural.

## Tipos de hogar (v1.4)

**Pregunta:** el paper estima Tb por tipo (Tabla 10) y Ts común (Tabla 5) sin precios de
usados, con errores estándar de ~0.02. ¿Por qué con un tipo no se identifican?

**Hipótesis:** subir Tb en δ, bajar Ts en δ y bajar P en δ/mu deja igual toda
transacción de usados. P es común a todos los tipos, pero mu varía por tipo (Tabla 7):
el ajuste δ/mu que compensa a un tipo no compensa al otro. Si mu es distinta entre tipos,
la dirección se rompe.

### Implementación
- `params.PAPER_TYPES`: los 4 tipos "Low WD" con mu, u0, u1, Tb común y Tb "no car"
  (Tablas 7-10). `GTypes(names, f)` elige tipos y fracciones. El paper no reporta f;
  se usa mitad y mitad.
- **Por tipo:** mu, u0, u1, tc_buy, tc_buy_nocar. **Comunes:** tc_sell,
  tc_sell_inspect, sigma_sell, accidentes (como el paper).
- `theta.Economy(g, th, f)`; `equilibrium.py` resuelve z = (EV_0, ..., EV_{T-1}, P) con
  D y S sumados sobre tipos ponderados por f. Con T = 1 da exactamente lo mismo que
  antes (las pruebas dan los mismos números).
- El tipo se observa. Las celdas llevan `tipo` y cada una usa las probabilidades de su
  tipo.
- Gradiente implícito con 27 parámetros contra diferencias finitas: 2e-8.

```
python main.py --reps 0:50 --N 40000 --types low_couple_poor,low_single_poor -v
python main.py --summarize --N 40000 --types low_couple_poor,low_single_poor
```

### Diseño
Los tres diseños usan N = 40,000 hogares en total, K = 10 y a_max = 25, con 50
réplicas y todos los parámetros libres.

| diseño | tipos | mu | parámetros |
|---|---|---|---|
| T1 | Couple Poor | 0.1131 | 18 |
| A | Couple Poor + Single Poor | 0.1131 / 0.0941 | 27 |
| B (control) | Couple Poor + Couple Rich | 0.1131 / 0.1119 | 27 |

T1 tiene los mismos hogares que A y B, así que separa el efecto de la heterogeneidad
del de tener más datos.

### Resultados (verosimilitud parcial, la del paper; entre paréntesis, la completa)

| | T1 | A (mu distinta) | B (mu casi igual) |
|---|---|---|---|
| se de tc_buy (tipo 0) | 2.69 (1.65) | **0.158** (0.126) | 1.53 (1.08) |
| se de tc_sell | 2.69 (1.65) | **0.155** (0.123) | 1.53 (1.08) |
| se de mu (tipo 0) | 0.022 (0.014) | **0.0057** (0.0038) | 0.0125 (0.0091) |
| corr(tc_buy_t0, tc_sell) entre réplicas | −1.000 | −0.994 | −1.000 |
| RMSE de P (miles DKK) | 13.3 (7.0) | **1.09** (0.73) | 7.8 (4.9) |
| cobertura al 95% de tc_buy y tc_sell | 0.92 (0.96) | 0.92 (0.94) | 0.98 (0.94) |

Sesgo despreciable en los tres diseños.

### Lectura

1. **Se confirma la hipótesis.**
   - Con dos tipos de mu distinta (A), el se de tc_buy y tc_sell baja 17 veces (2.69
     -> 0.16) y el error en precios pasa de 13 a 1.1 mil DKK, sin precios observados.
   - Con dos tipos de mu casi igual (B), casi no mejora: se de 1.5 y RMSE de P de 7.8.
     Sigue siendo la misma dirección, con correlación −1.000.
   - **Lo que identifica no es tener más tipos, sino que difieran en mu.** Así el
     paper puede estimar Tb y Ts por separado: sus 8 tipos tienen mu de 0.092 a 0.116.
2. **Sigue siendo la dirección más débil** (corr −0.99 en A), pero ya con precisión
   útil. El paper reporta se de ~0.02 para Tb: tiene 8 tipos, 4 marcas y ~39 millones
   de observaciones.
3. **El chatarreo endógeno se sigue separando de los accidentes** con la verosimilitud
   parcial: fracción voluntaria 0.352 contra 0.352 verdadera en A.
4. **Para la tesis:** si el modelo con reparación tiene un solo tipo de hogar, hereda
   el problema. Opciones:
   - tipos de hogar con mu distinta (lo del paper, si los datos traen tipos);
   - fijar tc_buy;
   - precios de usados como dato.

### Advertencias
- **El criterio de "mismo óptimo" es estricto.** Exige |ΔLL| < 1e-4 con LL de ~2e5. En
  ~30% de las réplicas uno de los tres arranques no lo cumple (incluido T1), en parte
  porque L-BFGS llega al tope de 500 iteraciones en la dirección plana. No guardé las
  estimaciones de los arranques perdedores, así que no sé si son otros máximos o el
  mismo mal pulido.
- **Convergencia:** B reporta convergencia de BHHH en 96-98% de las réplicas. A y T1,
  en 100%.
- **Las fracciones f = (0.5, 0.5) son supuesto.** El paper no las reporta.
