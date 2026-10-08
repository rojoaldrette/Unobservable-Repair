# Inventario del repo (2026-10-07): qué sirve y qué queda como historial

`claude/niu/` es el punto de partida nuevo. **No se borró nada.** Lo de abajo dice qué se
rescata de lo anterior y qué ya no se usa.

## Se rescata (ideas o resultados que siguen valiendo)

| qué | dónde | por qué |
|---|---|---|
| Réplica validada de Gillingham y su MC (100 réplicas, sin sesgo, cobertura ~0.95) | `claude/scripts/gillingham/`, `claude/output/estimaciones/gillingham/gill_propios_*` | prueba que el estimador DNFXP funciona; `niu/gillingham/` la reescribe |
| Tb y Ts solo se separan con tipos de mu distinta | `claude/docs/gillingham.md` (sec. "Tipos de hogar") | decide el diseño: 2 tipos con mu distinta |
| Identificación con Hu & Xin: teorema vs. Remark 5, mezcla por h, R como variable excluida | `claude/docs/id_modelo_vsHu.md` | base de la sec. de identificación del plan |
| Reparar como etapa 2 después de comerciar (p(r \| h) igual para keepers y compradores) | `claude/docs/estructura_decision.md` | da la ec. 6.4 de Hu & Xin con observabilidad parcial |
| Escalera de datos D0 (r observada) y D1 (r no observada); D2-D4 para después | `claude/docs/datos_ideales.md` | diseño del MC |
| s en log-odds con la logit de accidentes de Gillingham como caso sin reparar | `claude/docs/calibracion.md` | anida Gillingham; se retoma como "desviación" en el plan |
| Información asimétrica con un precio por (j, a) | `claude/docs/info_asimetrica.md` | alternativa que baja la dimensión de P |
| Opción 2 (reparar no cambia el accidente de este año) | `opcion_1_reparar.md`, `claude/docs/discusion_repair_o1.md` | sin selección por supervivencia en la mezcla |
| Chatarreo endógeno (sin él, precios negativos en edades altas) | `claude/docs/chatarreo_endogeno.md` | va en el modelo nuevo, igual que en Gillingham |
| Papers | `docs/papers/` | |

## Historial: no se usa en lo nuevo

| qué | por qué se deja |
|---|---|
| `claude/scripts/modelo_fin/` completo (n_s = 100, P por celda (j, a, s), Newton-Krylov, 13 regímenes, 30 parámetros) | no es corrible: la réplica 0 de la fase 1 no terminó en más de 12 h y el MC habría tardado semanas. Ver `niu/docs/modelo_fin.md` |
| `claude/scripts/correr_gpu.sh`, `slurm_estimar.sh`, `modelo_fin/slurm_mc.sh` | orquestan `modelo_fin` |
| `claude/scripts/analisis/`, `claude/scripts/comparacion/compare.py` | leen las salidas de `modelo_fin`; se reescriben en `niu/analisis` y `niu/comparacion` cuando exista `modelo_tesis` |
| `claude/scripts/gillingham/ED.py`, `main.py`, `montecarlo.py` viejo | reemplazados por `niu/gillingham/` |
| Calibraciones `tesis_v0`, `gill_s`, `defaults` | resultados viejos (v1.0-v1.7) |
| `claude/docs/manual.md`, `codigo.md`, `reporte_estimacion.md`, `newton_krylov.md`, `modelo_fin.md`, `version_hist.md` | documentan el código viejo |
| `claude/docs/fases_tesis.md`, `idea_km.md`, `inspeccion_zigzag.md`, `pruebas_robustes.md`, `datasets_montecarlo.md` | propuestas: se rescatan si el plan las necesita |
| `claude/output/estimaciones/modelo_fin/*` | equilibrios verdaderos de un diseño que se abandona |
| `scripts/main/` (Rust) | código del autor, no se toca |

## Hallazgos de la revisión que cambian decisiones

1. **La lectura de la Tabla 10 que se usaba ("suma") da 3% de hogares sin coche contra
   40% en el paper.**
   - La lectura "reemplaza" da 34%, precios que caen 13% al año (como reporta el paper)
     y usados de 1 año más baratos que el nuevo.
   - Con "suma", el usado de 1 año salía más caro que el nuevo, tanto en la réplica vieja
     como en `modelo_fin`.
   - Detalle en `niu/docs/gillingham.md`, sec. 2.
2. **En `modelo_fin` el costo de cómputo no venía del GPU sino del diseño:**
   - 7,200 precios por año;
   - 30 resoluciones con GMRES por año en cada gradiente;
   - L-BFGS de hasta 500 iteraciones × 3 arranques × 2 estimadores, sin log intermedio.
