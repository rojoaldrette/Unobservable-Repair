# Cuánto vale cada dato: diseños de Monte Carlo

Idea: organizar el Monte Carlo como una escalera de bases de datos y medir cuánto se
recupera de la reparación no observada en cada escalón. El resultado le dice a quien
tenga datos (registros, inspecciones, aseguradoras) qué le hace falta, y sigue valiendo
cuando lleguen datos reales.

Relacionado: `fases_tesis.md` (fases 3, 4 y 6), `verosimilitud_estructural.md`,
`chatarreo_endogeno.md`, `idea_km.md`, `info_asimetrica.md`.

---

## 1. Un solo proceso generador, muchas "bases"

Se simula **una vez** el panel completo desde el equilibrio (todo: x, o, h, r, ℓ, ℓ',
salida, motivo de la salida, precios). Cada diseño **borra o degrada** columnas de esa
misma simulación. Así las diferencias entre diseños son solo por la información, no por
el azar de la simulación.

Proceso generador base: a_max = 25, ℓ en log-odds, dos tipos de hogar (Couple Poor y
Single Poor), parámetros de Gillingham, R(j, a) bajo / medio / alto por marca, K años con
R_t distinto por año, chatarreo endógeno encendido.

## 2. La escalera

**Seguros (decisión del 2026-10-05):** D0 y D1.

| diseño | se observa | se borra | análogo real |
|---|---|---|---|
| **D0 oráculo** | propiedad por coche, ℓ cada año, R_t, **r** | precios | cota superior (no existe) |
| **D1 base** | propiedad por coche, ℓ cada año, R_t | **r**, precios, motivo de salida (solo "salida") | como Gillingham + el estado ℓ por coche (registro + inspección anual) |
| D2 ℓ con ruido | lo de D1 con ℓ_obs = ℓ + ν | ℓ verdadero | defectos de inspección, siniestros, odómetro como proxy |
| D3 ℓ cada dos años | lo de D1, ℓ solo en años de inspección | ℓ en años sin inspección | inspección bianual (syn en Dinamarca desde los 4 años) |
| D4 agregado tipo danés | celdas (tipo, j, a, t): CCPs, salidas, holdings | ℓ y r por coche | datos de Gillingham (fase 6) |

**Fuera de esta tesis:**
- reparación observada en una submuestra (facturas de taller, garantías): otro paper;
- precios de usados y motivo de salida como datos extra (descartado el 2026-10-06: con
  dos tipos de mu distinta los precios aportan poco a tc, y s(ℓ) ya da la tasa de
  accidentes).

Ejes que se cruzan con cualquier diseño (los "diales" de cantidad):
- **N** hogares (20k, 40k, 100k);
- **K** años y **dispersión de R_t** (spread chico o grande): cuánta variación en la
  variable excluida hace falta;
- **tipos de hogar** (1 o 2): con 1 tipo, tc_buy y tc_sell no se separan (v1.4).

## 3. Qué se mide en cada diseño

1. **Reparación:**
   - sesgo y RMSE de s_repair;
   - error medio en p(r | h), ponderado por q (no todas las celdas pesan igual);
   - tasa de reparación agregada por edad, que es lo que un ministerio querría saber.
2. **Estructurales:** mu, sigma_repair, tc_buy, tc_sell, u_s.
3. **Un contrafactual**, que es lo que al final importa: subir R un 20% (o un subsidio a
   la reparación) y medir el cambio en
   - edad media del parque,
   - hogares sin coche,
   - chatarreo y accidentes.

   Se calcula con θ̂ de cada réplica y se compara con el verdadero. Un dato "vale" si
   mejora el contrafactual, no solo los parámetros.

## 4. Qué estimador usa cada diseño

| diseño | estimador | ¿existe hoy? |
|---|---|---|
| D0 | ML estructural con r observada: F_r directa en vez de la mezcla | parte (oráculo de la primera etapa en `loglikelihood.py`) |
| D1 | ML estructural con mezcla sobre r (`verosimilitud_estructural.md`) | no (Fase 4) |
| D2 | ℓ latente con proxy: integrar ℓ con un filtro (forward algorithm) por coche | no; el más caro |
| D3 | transición de dos pasos: mezcla sobre (r_t, r_t+1), cuatro componentes (o dos si la reparación solo pasa en la inspección, `idea_km.md`) | no |
| D4 | momentos o ML por celdas integrando ℓ con q(ℓ \| j, a) (fase 6) | no |

## 5. Preguntas que contesta cada comparación

- **D0 contra D1: el costo de no ver r.** Es la pregunta central de la tesis: el
  resultado de Hu & Xin aplicado a un mercado en equilibrio.
- **D2 con σ_ν creciente: hasta qué ruido en la medición de ℓ sirve el método.**
- **D3: cuánto se pierde con inspección bianual.**
- **D4 contra D1: qué se pierde al agregar** (el experimento puente de la fase 6).

## 6. Orden sugerido

1. D0 y D1: salen directo de la Fase 4 del plan.
2. D3 y D2: requieren estimadores nuevos.
3. D4: la fase 6.

Todo esto corre en GPU. En local solo pruebas de humo (n_s chico, a_max chico, N chico).

## 7. Notas

- **El supuesto clave en todos los diseños es la opción 2** (reparar no cambia el
  accidente de ese año). Con la opción 1, las salidas informan sobre r y se enredan con
  el chatarreo.
- **La información asimétrica** (`info_asimetrica.md`) es otro proceso generador, no
  otra base. Se puede repetir la escalera con ella después.
