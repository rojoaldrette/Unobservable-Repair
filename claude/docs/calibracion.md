# Calibración de la tesis (`modelo_fin/calibracion.py`, "tesis")

La calibración con la que se corre todo (default desde v1.8).  Idea: **los mismos
parámetros que Gillingham**, para que la única diferencia entre los dos modelos sea poder
reparar.

**Criterio (pendiente explícito):** solo los parámetros tienen que coincidir con los de
Gillingham, **no los resultados** (precios, distribución, chatarreo, ...).  Las
diferencias en resultados entre G25 y M25 son el sesgo que se quiere mostrar; no se
calibra nada (s_repair, s_sigma, sigma_repair, R) para que desaparezcan.

## Valores

| parámetro | valor | fuente |
|---|---|---|
| a_max | 25 | Gillingham |
| mu, u0, u1, tc_buy, tc_buy_nocar | por tipo, Tablas 7-10 | Gillingham |
| tc_sell, tc_sell_inspect | 0.9106, 2.1929 | Tabla 5 (lectura raw) |
| p_new | (142.33, 110.09, 275.84) | Tabla 3 |
| p_scrap, beta | 2, 0.95 | no se reportan (propios) |
| u_s | 0 | Gillingham no tiene desutilidad de s |
| estado de desgaste | ℓ = logit(s), s = prob. de descomponerse | propio |
| ℓ de un coche nuevo | acc_int_j = (−5.62, −6.04, −5.67) | Tabla 4 |
| ℓ' = ℓ + acc_age_j − s_repair r + η | acc_age_j = (0.180, 0.222, 0.202) | Tabla 4 |
| s_repair | 0.5 (reparar baja las odds ~40%) | propio |
| s_sigma (sd de η) | 0.15 | propio |
| grid de ℓ | [−9, 1], n_s = 100 (paso 0.1): s de 1e-4 a 0.73 | propio |
| R(j, a), a = 1 | (4, 6.5, 10) mil DKK, +6% lineal por año | propio ("realista": ligero gasolina bajo, diésel medio, pesado alto) |
| sigma_repair | 0.3 | propio |
| chatarreo endógeno | encendido, sigma_sell = 0.3454 | Tabla 5 |
| tipos de hogar (estimar.py) | pareja pobre + soltero pobre, mitad y mitad | Tablas 7-10; fracciones propias |

**Por qué log-odds:** sin reparar y sin ruido, s(j, a) = sigmoid(acc_int_j + acc_age_j a)
es exactamente la logit de accidentes de Gillingham (Tabla 4).  El grid uniforme en ℓ da
resolución donde importa (s de 0.003 a 0.3 en 25 años) y η sigue siendo aditivo y normal
(Hu & Xin, Assumption 2).

**Por qué chatarreo:** sin él, a a_max = 25 un coche viejo solo sale vendiéndose y sus
precios quedan negativos (hasta −18 mil DKK).  Con él, el chatarreo pone un piso de
~(mu p_scrap + Ts)/mu ≈ 10-20 mil DKK, como en Gillingham.

## Resultado del equilibrio (un tipo, n_s = 40; `python compare.py`)

| | G25 (Gillingham) | M25 ("tesis") | M25 sin chatarreo |
|---|---|---|---|
| hogares sin coche | 0.016 | 0.015 | 0.016 |
| P light brown, edad 1 / 10 / 15 / 24 | 149.9 / 46.0 / 10.8 / 16.0 | 151.9 / 48.0 / 11.1 / 14.1 | 149.2 / 37.1 / −7.8 / −9.2 |
| chatarreo endógeno, edad 15 / 20 / 24 | 0.05 / 0.36 / 0.60 | 0.05 / 0.36 / 0.65 | 0 |
| prob. de descomponerse, edad 10 / 20 | 0.021 / 0.118 | 0.010 / 0.036 | 0.012 / 0.053 |
| Pr(reparar), edad 1 / 10 / 20 | — | 0.33 / 0.12 / 0.04 | 0.27 / 0.08 / 0.03 |

Con dos tipos (estimar.py): converge en 33 s (n_s = 40).  Soltero pobre: 2.7% sin coche,
coches más viejos y más reparación que pareja pobre (1.6%).

## Cómo cambiarla

```python
from calibracion import calibracion
import dataclasses
g = calibracion("tesis", a_max=25, n_s=100)
g = dataclasses.replace(g, s_repair=0.8, sigma_repair=0.5)   # otra calibración
```

En la línea de comandos: `--calib` (tesis, tesis_v0, gill_s, defaults), `--a_max`, `--n_s`.
Las calibraciones viejas quedan para reproducir resultados de v1.0-v1.7.
