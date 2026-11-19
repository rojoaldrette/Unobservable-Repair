# Cómo correr el Monte Carlo en la workstation

Modelo de `docs/tex/tesis.tex` (código en `claude/niu/final/`), 1000 réplicas en 2 GPUs.
Todo se corre desde la raíz del repositorio con `claude/niu/final/correr_gpu.sh`.

| Paso | Qué hace | Tiempo estimado |
|---|---|---|
| 0. Actualizar | `git pull` | — |
| 1. Pruebas | tests y prueba de humo | 10–20 min |
| 2. Equilibrio verdadero | resuelve la economía y la revisas | 5–15 min |
| 3. Prueba real | 2 réplicas a tamaño completo, para medir tiempos | ~30 min |
| 4. Corrida larga | 1000 réplicas | **~2.5–3 días** |
| 5. Cierre | faltantes, resumen y perfiles por edad | 2–4 h |

Los tiempos salen del MC de octubre (~8 min por réplica por GPU). La prueba real del paso 3
da el número exacto para esta especificación.

---

## 0. Actualizar

```bash
cd ~/Documents/Unobservable-Repair
git pull
```

Usa el mismo entorno de Python (con JAX en GPU) que en octubre. Para comprobar que JAX ve las GPUs:

```bash
python -c "import jax; print(jax.devices())"     # deben salir 2 dispositivos cuda
```

## 1. Pruebas (¿funciona todo?)

```bash
bash claude/niu/final/correr_gpu.sh pruebas
```

Corre, en orden:
- las pruebas de Gillingham;
- las pruebas del modelo;
- la prueba de humo del MC;
- la prueba de humo de `por_edad_mc`.

**Todo debe decir `[ok]` y terminar sin `Traceback`.** En la prueba de humo es normal que
diga `convergió = False`, porque corta el optimizador en 20 iteraciones. Si algo falla, para
y mándame la salida.

## 2. Equilibrio verdadero

```bash
bash claude/niu/final/correr_gpu.sh verdad
```

Escribe `claude/niu/final/output/modelo_tesis/teoria/R_LB/`. Este equilibrio lo usan todas
las corridas que siguen. Al final imprime un resumen por régimen y una tabla por edad.
Revisa tres cosas:

1. **`masa_borde_w` < 0.005** en los tres regímenes (`resumen.csv`). Si es mayor, el grid de
   $w$ se queda corto: avísame antes de seguir.
2. **Reparación en las dos marcas** (`por_edad.csv`, columna `reparar`). Heavy brown (`j = 1`)
   ahora tiene precio de reparación constante (16.6); debe tener reparación razonable, ni
   ~0 ni ~1 en todas las edades.
3. **La reparación baja al subir el precio**: en `resumen.csv`, `reparacion` del régimen 0 >
   régimen 1 > régimen 2.

```bash
column -s, -t < claude/niu/final/output/modelo_tesis/teoria/R_LB/resumen.csv | less -S
```

## 3. Prueba real: 2 réplicas a tamaño completo

```bash
TAG=prueba REPS=2 bash claude/niu/final/correr_gpu.sh mc
```

Corre la réplica 0 en la GPU 0 y la 1 en la GPU 1, y guarda en `montecarlo/prueba/` (no se
mezcla con la corrida final).

- La **réplica 1** da el tiempo normal por réplica. Busca la línea `rep 1 lista (XXX s)` en
  `claude/niu/final/output/logs/montecarlo_1-1.log`.
  **Tiempo de la corrida larga ≈ (segundos de la réplica 1) × 500 / 3600 horas**: cada GPU
  hace 500 réplicas.
- La **réplica 0** tarda más (~30 min), porque además arranca desde dos puntos perturbados
  para revisar que no haya máximos locales.

Revisa en los logs que cada diseño diga `convergió = True` y en
`montecarlo/prueba/resumen_reps*.csv` que `convergio` sea `True`. También que exista
`inferencia_resumen_reps*.csv`.

## 4. Corrida larga: 1000 réplicas

Dentro de `tmux`, para que siga corriendo al cerrar ssh:

```bash
tmux new -s mc
REPS=1000 bash claude/niu/final/correr_gpu.sh mc
# salir sin detenerlo: Ctrl-b y luego d
```

Reparte las réplicas 0–499 a la GPU 0 y 500–999 a la GPU 1. Cada réplica:
1. simula el panel;
2. estima los diseños 1–3 en la GPU y los dos de Gillingham en CPU al mismo tiempo;
3. calcula la inferencia de la reparación;
4. agrega filas a los CSV.

**Seguimiento:**

```bash
tmux attach -t mc                                  # volver a la sesión
bash claude/niu/final/correr_gpu.sh estado         # GPUs, procesos, estimaciones terminadas, disco
tail -f claude/niu/final/output/logs/montecarlo_0-499.log
```

Cada 5 réplicas el log imprime `[avance] i/500 réplicas, X h transcurridas, ~Y h restantes`.

**Si se corta** (reinicio, corte de luz): una réplica se escribe completa al terminar, así
que solo se pierde la que estaba en curso. Busca la última `rep N lista` de cada log y
relanza cada GPU desde la siguiente (ejemplo: la GPU 0 llegó a la 312 y la GPU 1 a la 805):

```bash
cd claude/niu/final/modelo_tesis
CUDA_VISIBLE_DEVICES=0 nohup python -u montecarlo.py --reps 313:500  > ../output/logs/montecarlo_313-499.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u montecarlo.py --reps 806:1000 > ../output/logs/montecarlo_806-999.log 2>&1 &
```

Los archivos nuevos llevan su propio rango en el nombre (`reps313-499`). Al leerlos, junta
todos los `*_reps*.csv`. Si una réplica quedó a medias en Gillingham, puede salir repetida en
`gill_*`: quédate con la última.

## 5. Cierre

```bash
REPS=1000 bash claude/niu/final/correr_gpu.sh faltantes   # re-estima las (rep, diseño) que fallaron
bash claude/niu/final/correr_gpu.sh estado                 # confirmar 3000 estimaciones
bash claude/niu/final/correr_gpu.sh resumen                # resumen_mc.csv
REPS=1000 bash claude/niu/final/correr_gpu.sh por_edad     # perfiles por edad en cada θ̂ (~1.5–3 h)
```

- `faltantes` no re-estima Gillingham, y sus archivos llevan el sufijo `_faltantes`.
- `por_edad` se salta lo que ya está. Si se corta, vuelve a correr el mismo comando.

## Qué queda y dónde

Todo en `claude/niu/final/output/modelo_tesis/montecarlo/base/`. Las columnas, ejemplos y
gráficas posibles están en `claude/niu/final/docs/bases_datos.pdf`. Archivos nuevos respecto
a ese documento:

- `inferencia_reparacion_reps*.csv.gz`: celdas de inferencia de $r$, **solo réplicas 0–99**
  (con 1000 serían ~120 millones de filas). Se cambia con
  `python -u montecarlo.py ... --inferencia_hasta N`.
- `inferencia_resumen_reps*.csv`: **todas** las réplicas. Una fila por
  (rep, estimador, info) con `n, tasa_real, media, brier, auc`; `info` es `prior`, `post_d2`
  o `post_d3`. Con esto salen las distribuciones del AUC y el Brier en las 1000 réplicas.

Tamaño esperado: menos de 1 GB en total.

**Para subir los resultados** (commit desde la workstation): los logs ya están ignorados.
Antes de agregar, revisa que ningún archivo pase de 100 MB, que es el límite de GitHub:

```bash
find claude/niu/final/output -size +90M
```

Si `inferencia_reparacion_reps0-499.csv.gz` pasa del límite, no lo subas por git: cópialo
con `scp`. No subas `teoria/R_LB/panel.csv.gz` si es grande; se regenera con la semilla.
