# Newton-Krylov: qué es y por qué lo usamos

## El problema

El equilibrio es un vector z que cumple F(z) = 0. En `modelo_fin` con dos tipos:

    z = (EV_0, EV_1, P)                                   N ≈ 21,600 incógnitas
    F(z) = [ EV_0 − T_0(EV_0, P) ;  EV_1 − T_1(EV_1, P) ;  log D(EV, P) − log S(EV, P) ]

## Newton

Desde z_k, linealizar y despejar:

    J(z_k) d = −F(z_k),       z_{k+1} = z_k + d          (J = dF/dz, N x N)

Converge de forma cuadrática: el error pasa de 1e-2 a 1e-4 a 1e-8 a 1e-16. Con un buen
arranque son 3-6 pasos. Es lo que se hizo en la réplica de Gillingham (224 incógnitas).

**El problema es J.** Con N = 21,600:
- guardarla densa son N² · 8 bytes ≈ 3.7 GB;
- factorizarla (LU) cuesta ~N³ ≈ 10¹³ operaciones por paso de Newton.

## La parte "Krylov": resolver J d = −F sin armar J

GMRES (un método de Krylov) resuelve J d = b usando **solo productos J · v**:

1. Construye el espacio b, J b, J² b, ..., J^m b (el "subespacio de Krylov").
2. Busca la combinación de esos vectores que hace más chico ||J d − b||.
3. Si la matriz está bien condicionada, en unas decenas de productos ya tiene d con
   error de 1e-10.

**Cómo se calcula J · v sin J:** con diferenciación automática en modo forward
(`jax.jvp`). Calcula la derivada direccional de F en dirección v al costo de ~2-3
evaluaciones de F. JAX la saca exacta (no son diferencias finitas).

Así, un paso de Newton cuesta ~(número de iteraciones de GMRES) × (costo de F), y en
memoria nunca hay nada de N x N.

## Precondicionador

GMRES converge rápido si J se parece a la identidad. Un precondicionador es una
aproximación barata de J⁻¹ que se aplica antes:
- bloque EV: J ≈ I − beta M, con autovalores lejos de cero (≥ 1 − beta = 0.05), así que
  ya está bien condicionado;
- bloque P: d(log D − log S)/dP ≈ −mu/sigma_s en la diagonal (es lo que usa hoy el
  tâtonnement de `ED.py`). Se usa su inversa.

## Qué hace hoy `ED.py` y qué cambia

- **Hoy:** Newton-Krylov solo sobre P (7,200 incógnitas con a_max = 25). Para cada P
  que prueba, resuelve la Bellman completa (ciclo anidado).
- **Propuesta:** Newton-Krylov sobre z = (EV, P) conjunto, como el Newton denso de la
  réplica de Gillingham. EV y P se ajustan a la vez y no hay ciclo anidado.

## El gradiente implícito usa lo mismo

Para estimar hace falta dz/dθ = −J⁻¹ F_θ (ver `verosimilitud_estructural.md`). Son
sistemas lineales con la misma J: se resuelven con GMRES y productos J · v (jvp) o
J^T · w (`jax.vjp`).

## Por qué esto es bueno para GPU

Todo son operaciones de arreglos (evaluar F, jvp, vjp, productos punto de GMRES) sin
ramas en Python. `jax.jit` compila todo el paso una vez y la GPU lo corre en paralelo.
No hay factorizaciones densas grandes, que son lo que más memoria pide.
