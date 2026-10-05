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
| `tests.py` | `python tests.py [--a_max 7]` |
| `../comparacion/compare.py` | corre G25, G7, M7 y M7_gill y escribe `comparacion/output/*.csv` |

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
