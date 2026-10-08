# Réplica de Gillingham et al. (`claude/niu/gillingham/`)

Gillingham, Iskhakov, Munk-Nielsen, Rust & Schjerning (2019 WP), "Equilibrium Trade in
Automobiles" (`docs/papers/`). Equilibrio estacionario del mercado de autos nuevos y
usados (sec. 3-4) y su estimador de máxima verosimilitud DNFXP (sec. 5.1, apéndice D),
con los parámetros que estimaron (apéndice F). Versión limpia y desde cero de
`claude/scripts/gillingham/`, que queda como historial.

**Diseño por defecto:** 2 marcas (light brown y heavy brown), 2 tipos de hogar (Low WD
Couple Poor y Low WD Single Poor, mitad y mitad), a_max = 25, beta = 0.95.

## 1. Modelo

**Estado al inicio del año:** coche (j, a), con a = 1..24 activo y a = 25 terminal, o
"sin coche".

**Elecciones de un dueño de (i, a)** (figura 1, nidos de marca y edad colapsados):

- **purge:** se deshace del coche y se queda sin coche;
- **keep:** se queda el coche;
- **trade:** se deshace del coche y compra (j, d), con d = 0 (nuevo) o d = 1..24.

Al deshacerse de un coche activo elige entre vender (cobra mu P(i, a) − Ts(a)) y
chatarrear (cobra mu p_scrap). Tiene escala sigma_sell y es una decisión estática
(ec. 18). Quien está en la terminal chatarrea a la fuerza y elige entre purge y trade.
Quien no tiene coche elige entre seguir así y comprar.

**Utilidad del año:** u(j, d) = u0_j + u1_j d (u2 = 0) para el coche que se maneja.
"Sin coche" vale 0. Comprar cuesta mu × precio + Tb.

**Física:** después de manejar, la tenencia (j, d) envejece a (j, d+1) o se accidenta
(pasa a la terminal) con

    alpha(j, d) = logit^-1(acc_int_j + acc_age_j d)      (Tabla 4)

La edad 24 llega a la terminal con certeza.

**Equilibrio (Teorema 4):** precios P(j, a), a = 1..24, comunes a todos los tipos, tales que

- la demanda agregada iguala a la oferta, D(j, a) = S(j, a) (ecs. 21-22, sumadas sobre
  tipos con pesos f_t);
- cada tipo está en su distribución estacionaria, q_t = q_t Ω_t(P) Q.

Lo chatarreado no se ofrece. Los nuevos se venden a p_new fijo con oferta infinitamente
elástica.

## 2. Parámetros (`params.py`)

Todos los números vienen de las tablas del apéndice F. El código trae las 4 marcas y los
8 tipos; `Config(brands=..., types=...)` elige cuáles se usan.

| parámetro | valor (marca LB, HB) | fuente |
|---|---|---|
| mu | 0.1131 (pareja pobre), 0.0941 (soltero pobre) | Tabla 7 |
| u0 | pareja (3.649, 5.154); soltero (2.404, 3.582) | Tabla 8 |
| u1 | pareja (−0.146, −0.220); soltero (−0.098, −0.160) | Tabla 9 |
| Tb "common", Tb "no car" | pareja 6.594, 1.790; soltero 6.546, 3.082 | Tabla 10 |
| sigma_sell, Ts, Ts en año de inspección | 0.3454, 0.9106, 2.1929 | Tabla 5 (lectura raw) |
| acc_int, acc_age | (−5.625, −5.673), (0.180, 0.202) | Tabla 4 |
| p_new | (142.33, 275.84) mil DKK | Tabla 3 |
| u2, dummy de edad par, escalas de marca y edad | 0, omitido, 1 | no se reportan |
| p_scrap, beta, fracciones f | 2 mil DKK, 0.95, (0.5, 0.5) | no se reportan (propios) |

**Comprobación de unidades:** u0 / mu = 3.649 / 0.1131 = 32.3 mil DKK por un año de coche
nuevo. Es el número que da el texto (sec. 6.2) para la pareja pobre; para la pareja rica,
4.032 / 0.1119 = 36.0, también como en el texto.

### Lecturas dudosas de las tablas, y cuál se usa

El WP tiene problemas de formato: los errores estándar de la Tabla 5 son iguales a las
estimaciones y las etiquetas se desalinean con `pdftotext -layout`. Se probaron las
lecturas posibles contra hechos que reporta el paper:

- 40% de hogares sin coche (figura 4);
- precios que caen ~13-14% al año (sec. 6.2);
- más compras de usados que de nuevos (figura 6).

| lectura de la Tabla 10 | sin coche | compran nuevo / usado al año | keep, edades 1-10 |
|---|---|---|---|
| Tb en utils; sin coche paga **Tb + Tb_nc** ("suma", la de `scripts/`) | 0.03 | 0.055 / 0.031 | 0.96 |
| Tb en utils; sin coche paga **solo Tb_nc** ("reemplaza", **la que se usa**) | 0.29-0.34 | 0.04 / 0.19 | 0.5-0.8 |
| Tb en miles de DKK (resta mu Tb) | 0.01-0.03 | — / 0.88 | 0.05 |

(Corrido con J = 2 y los dos tipos pobres; los rangos son por par de marcas.)

- **"En DKK" queda descartada:** con ella casi nadie se queda su coche.
- **"Suma" contradice el 40% sin coche.** Es la lectura que usaba `claude/scripts/`.
- **"Reemplaza" es la única cerca de los tres hechos:**
  - 34% sin coche con LB+HB;
  - precios que caen 13.1% (LB) y 12.6% (HB) al año entre las edades 1 y 15;
  - más compras de usados que de nuevos;
  - el precio de un usado de 1 año queda debajo del nuevo (139 contra 142 y 258 contra
    276). Con "suma" quedaba encima, lo cual no tiene sentido.

  Interpretación: Tb "common" es el costo de comprar cuando se cambia de coche (o se
  sale de la terminal), y Tb "no car" es el de comprar cuando no se tenía coche.
- **Tabla 5:** se lee como sigma_sell = 0.3454, Ts = 0.9106 y Ts = 2.1929 en años de
  inspección (edad par >= 4). El −2.1929 se toma como el coeficiente en la utilidad de
  vender, así que es un costo. Con esta lectura aparece el zig-zag de la figura 7. **Sigue
  por verificar contra la versión publicada (JPE).**

`Config(nocar="suma")` y `Config(tc_units="dkk")` reproducen las otras lecturas.

### Equilibrio con los defaults

**Agregados:**

| | valor |
|---|---|
| sin coche | 0.336 (pareja 0.31, soltero 0.36) |
| compran nuevo / usado al año | 0.038 / 0.191 de la población |
| chatarreo endógeno / accidentes, por coche al año | 0.026 / 0.031 |
| edad media del parque | 9.6 años |
| participación por marca | LB 0.50, HB 0.14 |

**Por edad:**

| edad | 1 | 2 | 4 | 5 | 10 | 15 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|
| P light brown | 139.1 | 125.1 | 105.9 | 88.6 | 49.7 | 19.5 | 22.5 | 21.8 |
| P heavy brown | 258.0 | 234.3 | 196.9 | 171.4 | 96.2 | 39.0 | 26.5 | 23.5 |
| keep LB / HB | .70/.51 | .70/.51 | .82/.69 | .69/.50 | .82/.68 | .65/.46 | .55/.50 | .26/.19 |
| Pr(chatarrear) LB | 0 | 0 | 0 | 0 | 0 | .04 | .41 | .46 |

**Lectura:**

- El zig-zag de inspección aparece en keep: sube en las edades pares de inspección.
- Los precios de los coches viejos tienen un piso de ~(mu p_scrap + Ts)/mu. Ahí sube el
  chatarreo.

## 3. Código

| archivo | qué tiene |
|---|---|
| `params.py` | tablas del paper, `Config` (estático) y `theta(cfg)` (parámetros, dict de arreglos) |
| `utils.py` | layout X = [act(j,a) \| term(j) \| none], H = [used(j,d) \| new(j) \| none]; log-sumas; rebanar θ por tipo |
| `bellman.py` | u, alpha, Ts, Tb, valores por elección y operador Γ (un tipo) |
| `probabilidades.py` | CCPs, Q, M = ΩQ, q estacionaria, probabilidades de los 5 resultados observables |
| `equilibrio.py` | sistema conjunto F(z) con z = (EV_0, EV_1, P), Newton denso, estadísticas de mercado |
| `gen_dataset.py` | panel de hogares desde el equilibrio y agregación a celdas (ec. 40) |
| `ll_estim.py` | verosimilitud parcial (apéndice D) y completa, gradiente implícito, L-BFGS + BHHH |
| `montecarlo.py` | Monte Carlo del estimador y tabla resumen |
| `tests.py` | pruebas (abajo) |

**Convenciones:**

- `cfg` (Config) es estático en jit. θ es un dict de arreglos (dinámico): un θ nuevo no
  recompila.
- Los campos por tipo (`mu, u0, u1, u2, tc_buy, tc_buy_nocar`) llevan el tipo en el
  primer eje, y se usa `jax.vmap` sobre tipos.
- Todo es denso: n = 51 estados por tipo y 150 incógnitas en el equilibrio.

**Equilibrio:**

- Un solo sistema F(z) = 0 que junta las Bellman de los tipos y log D − log S.
- Arranque: precios con depreciación de 13% anual y 1,000 aproximaciones sucesivas.
- Luego Newton con jacobiano denso y búsqueda de línea.
- Tarda 4 s en frío y 3 ms en caliente. Es el mismo punto fijo que el doble Newton del
  paper (sec. 3.5).

## 4. Estimador (DNFXP)

**Datos:** panel de N hogares por K años desde la distribución estacionaria (tipo
observado). Se agregan a celdas h_{k−1} -> (x_k, o_k, h_k). Hay 5 resultados: keep,
purge (vende o chatarrea) y trade (vende o chatarrea).

**Verosimilitudes:**

- **"parcial" (la del paper):** no se ven ni accidentes ni precios. Pr(o, h' | h) =
  (Q C)[h, o] · buy(h')^{1(trade)}. Una salida del parque puede ser accidente o
  chatarreo endógeno.
- **"completa" (oráculo):** además ve el estado x, es decir, el accidente.

**Libres (27 con 2 tipos y 2 marcas):**

- por tipo: mu, u0_j, u1_j, Tb, Tb_nc;
- comunes: Ts, Ts de inspección, sigma_sell, acc_int_j, acc_age_j.

**Normalizaciones:** sigma = sigma_trade = 1 y u(sin coche) = 0. Fijos y conocidos: p_new,
p_scrap y beta.

**Gradiente:** función implícita, dz/dθ = −F_z⁻¹ F_θ. Salen los scores por celda, y de
ahí el gradiente y la matriz BHHH.

**Optimización:**

- L-BFGS: imprime cada 10 iteraciones la LL, el gradiente y cuántas veces se resolvió el
  equilibrio.
- Luego BHHH, que da los errores estándar por el método delta.
- Arranques: la verdad más `n_starts` perturbados; se guardan todos.

**Lo que ya se sabe de la versión anterior** (`claude/docs/gillingham.md`, 100 réplicas
con 3 marcas):

- sin sesgo y con cobertura de 0.92-1.00;
- ~10 s por estimación;
- **Tb y Ts solo se separan si hay tipos con mu distinta** (por eso los dos tipos por
  defecto tienen mu de 0.113 y 0.094).

## 5. Cómo se corre

```bash
cd claude/niu/gillingham
python -u tests.py                      # pruebas, a_max = 10 (segundos)
python -u tests.py --a_max 25           # tamaño del paper
python -u montecarlo.py --smoke         # humo del estimador (menos de un minuto)
python -u montecarlo.py --reps 0:100 -v # MC completo; salida en claude/niu/output/gillingham/<tag>/
python -u montecarlo.py --summarize --tag A25_N20000_K10_T2_p27
```

**Opciones:** `--N`, `--K`, `--brands`, `--types`, `--a_max`, `--nocar`, `--infos`,
`--fix tc_buy,...`, `--n_starts`, `--lbfgs_iter`, `--bhhh_iter` y `--print_every` (línea `[avance]` con tiempo transcurrido y restante; default cada 5 réplicas). La lista de salidas está
en el docstring de `montecarlo.py`.

**Verificado (2026-10-07, CPU, a_max = 10):**

- las CCPs y las filas de Q y M suman 1;
- q = qM;
- flujo estacionario por marca (Teorema 3);
- dT/dEV = βM (Lema L1);
- el gradiente implícito coincide con diferencias finitas a 1e-7 (parcial y completa);
- la prueba de humo del estimador converge al mismo óptimo desde los dos arranques.

**No corrido:** el Monte Carlo a tamaño completo. A juzgar por la versión anterior,
deberían ser unos minutos por réplica.

## 6. Diferencias con el paper (declararlas)

1. **Marcas y tipos:** 2 marcas y 2 tipos de hogar, contra 4 y 8 en el paper. Tampoco
   hay dummy de edad par en la utilidad, ni u2, ni bloque de manejo (sec. 6.1). La
   utilidad u(j, a) es la indirecta reducida, que es lo que reportan las Tablas 8 y 9.
2. **Nidos:** los de marca y edad se colapsan, porque el paper no reporta sus escalas.
3. **Lecturas de las tablas:** las de la sección 2, que siguen por confirmar con la
   versión publicada.
4. **Solver:** un Newton conjunto en (EV, P) en vez de dos Newton anidados. Es el mismo
   equilibrio.
5. **Datos:** son simulados del propio modelo. El paper no tiene Monte Carlo: estima con
   el registro danés.
