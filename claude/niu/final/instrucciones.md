# Cómo correr el Monte Carlo en la workstation

Modelo de `docs/tex/tesis.tex` (código en `claude/niu/final/`), 1000 réplicas en 2 GPUs.
Todo se corre desde la raíz del repositorio con `claude/niu/final/correr_gpu.sh`.

| Paso | Qué hace | Tiempo estimado |
|---|---|---|
| 0. Actualizar | `git pull` | — |
| 1. Pruebas | tests y prueba de humo | 10–20 min |
| 2. Equilibrio verdadero | **ya hecho y revisado (18/11)**: se puede saltar | (5–15 min) |
| 3. Prueba real | 2 réplicas a tamaño completo, para medir tiempos | ~30 min |
| 4. Corrida larga | 1000 réplicas | **~2.5 días** |
| 5. Cierre | faltantes, resumen y perfiles por edad | 2–4 h |

**Cambio desde la primera prueba (19/11):** el Gillingham "parcial" ya no se corre. Con estos
datos no identifica las preferencias (`docs/cambios_y_notas.md`, entrada del 19/11). Solo se
estima el Gillingham completo, que ve el motivo de salida.

**Los CSV se agregan, no se reemplazan.** Correr dos veces lo mismo en la misma carpeta
duplica filas. Por eso `pruebas` y `prueba` borran su carpeta antes de correr, y `mc` se
niega a empezar si ya hay resultados en esa carpeta (ver paso 4).

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

Borra `montecarlo/smoke/` y corre, en orden:
- las pruebas de Gillingham;
- las pruebas del modelo;
- la prueba de humo del MC;
- la prueba de humo de `por_edad_mc`.

**Todo debe decir `[ok]` y terminar sin `Traceback`.** En la prueba de humo es normal que
diga `convergió = False`, porque corta el optimizador en 20 iteraciones. Ahora debe aparecer
solo `Gillingham [completa]`, ya no `[parcial]`. Si algo falla, para y mándame la salida.

## 2. Equilibrio verdadero (ya hecho)

`claude/niu/final/output/modelo_tesis/teoria/R_LB/` ya está en el repositorio (corrida del
18/11) y pasó los chequeos:
- masa en los bordes del grid de 0.13% a 0.32%;
- la reparación baja al subir su precio: 0.38 > 0.27 > 0.16;
- heavy brown repara poco pero no cero.

Este cambio no lo afecta, así que **no hace falta volver a correrlo**. Si lo vuelves a correr
(`bash claude/niu/final/correr_gpu.sh verdad`), sale igual y sobrescribe la carpeta.

## 3. Prueba real: 2 réplicas a tamaño completo

```bash
bash claude/niu/final/correr_gpu.sh prueba
```

Borra `montecarlo/prueba/` (la prueba anterior, que tenía el Gillingham parcial) y corre la
réplica 0 en la GPU 0 y la 1 en la GPU 1. Al final imprime las últimas líneas de los dos
logs. Revisa:

1. **Los tres diseños convergen**: `rep 0 [diseño d]: ... convergió = True` y lo mismo para la
   réplica 1, en `claude/niu/final/output/logs/montecarlo_0-0.log` y `montecarlo_1-1.log`.
2. **Gillingham solo con `[completa]`** y `convergió = True`.
3. **Tiempo:** la línea `rep 1 lista (XXX s)` da el tiempo normal por réplica (la primera vez
   fueron 519 s). **Horas de la corrida larga ≈ XXX × 500 / 3600**, porque cada GPU hace
   500 réplicas. La réplica 0 tarda más (~30 min), porque además arranca desde dos puntos
   perturbados para revisar que no haya máximos locales.
4. Existen `montecarlo/prueba/inferencia_resumen_reps0-0.csv`, `..._reps1-1.csv` y la carpeta
   `montecarlo/prueba/inferencia_reparacion/` con `rep0000.csv.gz` y `rep0001.csv.gz`.

Si quieres que yo lo revise, súbelo (`git add claude/niu/final/output && git commit && git push`)
y avísame.

## 4. Corrida larga: 1000 réplicas

Dentro de `tmux`, para que siga corriendo al cerrar ssh:

```bash
tmux new -s mc
REPS=1000 bash claude/niu/final/correr_gpu.sh mc
# salir sin detenerlo: Ctrl-b y luego d
```

Reparte las réplicas 0–499 a la GPU 0 y 500–999 a la GPU 1. Guarda en `montecarlo/base/`.
Cada réplica:
1. simula el panel;
2. estima los diseños 1–3 en la GPU y, al mismo tiempo en CPU, el Gillingham completo;
3. calcula la inferencia de la reparación;
4. agrega filas a los CSV.

Si `montecarlo/base/` ya tiene resultados, el script se niega a empezar desde la réplica 0
para no duplicar. Ve a "Si se corta".

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
todos los `*_reps*.csv`. Si una réplica quedó a medias, su Gillingham puede salir repetido en
`gill_*`: quédate con el último.

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
gráficas posibles están en `claude/niu/final/docs/bases_datos.pdf`. Gillingham aparece solo
como `gill_completa`. Para la inferencia de la reparación:

- `inferencia_reparacion/rep0000.csv.gz`, …: celdas de inferencia de $r$, **un archivo por
  réplica, solo réplicas 0–99** (con 1000 serían ~120 millones de filas). Se cambia con
  `python -u montecarlo.py ... --inferencia_hasta N`. En R:
  `list.files(".../inferencia_reparacion", full.names = TRUE) |> map_dfr(read_csv)`.
- `inferencia_resumen_reps*.csv`: **todas** las réplicas. Una fila por
  (rep, estimador, info) con `n, tasa_real, media, brier, auc`; `info` es `prior`, `post_d2`
  o `post_d3`. Con esto salen las distribuciones del AUC y el Brier en las 1000 réplicas.

Tamaño esperado: menos de 1 GB en total (~1.6 MB por réplica de celdas de inferencia, ~165 MB
las 100).

**Para subir los resultados** (commit desde la workstation): los logs sí se suben; son chicos
y sirven para revisar. Ningún archivo debería pasar de 100 MB, que es el límite de GitHub,
pero conviene comprobarlo antes de agregar:

```bash
find claude/niu/final/output -size +90M          # no debe imprimir nada
```
