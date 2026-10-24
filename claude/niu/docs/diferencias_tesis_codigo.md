# Diferencias entre la tesis y el código

Comparación de `docs/tex/tesis.tex` (sección Modelo) contra `claude/niu/modelo_tesis/`
(2026-10-23). Ninguna diferencia cambia resultados; son cosas de redacción o decisiones del
código que la tesis todavía no refleja. Las líneas son de `tesis.tex` en esa fecha.

## 1. Cambio pendiente en el código (no en la tesis)

- **Costo de venta en años de inspección.** La nota al pie (l. 46) dice que no se usa la
  especificación de costos de venta de Gillingham; el código usa `tc_sell = 0.9106` y
  `tc_sell_inspect = 2.1929` (este último reemplaza al primero en edades pares ≥ 4,
  `inspect_age_min = 4`), lo que produce el zig-zag de precios y chatarreo.
  **Decisión:** quitar `tc_sell_inspect` y dejar un solo $T_s$ constante en la edad antes de
  los MC finales. Con eso la nota al pie queda bien como está.
- **Edad y edad² en la probabilidad de accidente.** Hoy
  $s=\text{logit}^{-1}(\text{acc\_int}_j+\text{acc\_age}_j\,d+w)$ (`bellman.breakdown_prob`).
  Pendiente: agregar un término $\text{acc\_age2}_j\,d^2$. Se hace junto con el cambio de
  $T_s$, antes de los MC finales. Ojo: la media de $w'$ usa $\delta_j-\text{acc\_age}_j$
  (`w_transition`) y la calibración $\delta_j=\text{acc\_age}_j+\kappa\bar p$ supone una
  curva lineal en la edad; con $d^2$ hay que redefinir ese drift (p. ej. restar la pendiente
  local $\text{acc\_age}_j+\text{acc\_age2}_j(2d+1)$, que depende de la edad).
- **Idea, no decidida:** edad² en la utilidad (hoy `u2 = 0` fijo).

## 2. Matrices y dimensiones

Versión corregida completa en `claude/niu/docs/tesis_transiciones.tex`. Resumen:

- Terminal $(j,\bar a)$ y nuevo $(j,0)$ son **un solo estado** por marca (no $W$).
  $n = J(\bar a-1)W+J+1$.
- l. 190: $P\in\mathbb{R}^{J(\bar a-1)W}$ (no $(J(\bar a-1)-2)W-1$).
- l. 194-196: $q\in\mathbb{R}^{n}$; el simplex es de dimensión $n-1$ (aparecen $+4$ y $+3$).
  La fórmula $(J\bar a-3)W+4$ solo es correcta con $J=3$.
- l. 198 (demanda): separar a los dueños de terminal, que no tienen $w$ (la suma sobre $x$
  no aplica a $d=\bar a$).
- $\Omega$: subbloques de tamaño distinto según origen/destino ($W\times W$, $W\times1$,
  $1\times W$, escalar). En $\Xi$ el último 0 de la diagonal sobra (quitaría el keep del
  último $w$); ese 0 va en $\Lambda_j$ en el lugar del terminal.
- $Q$: $1-s(w)$ va en una **diagonal** $\bar S$ (está dibujado como columna); columna
  terminal $\mathbf{s}$; edad $\bar a-1$ va toda a terminal; fila del nuevo
  $(1-s(w_{0j}))e'_{w_{0j}}$. "$Q_s$ de tamaño $\bar a\times1$" lista $\bar a+1$ bloques.
- $A$: $C_{ja}$ usa la reparación de la edad **$a-1$** (la de la tenencia, antes de
  envejecer); $C_{j1}$ es $f(\cdot\mid r=0)$ (nuevos no reparan); terminal es un 1, no un
  $C_{\bar a}$ de $W\times W$.
- Nota: $Q\cdot A$ como factorización es válida por la opción 2 (accidente no depende de $r$).
  Comprobado numéricamente: $Q\cdot A$ = kernel del código.

## 3. Especificación

| Tema | Tesis | Código | Qué hacer |
|---|---|---|---|
| Prob. de accidente (l. 60-61, 340) | $s=\text{logit}(w)$, igual entre marcas | $s=\text{logit}(\text{acc\_int}_j+\text{acc\_age}_j\,d+w)$, $w$ = desviación de la curva de Gillingham | Equivalentes con $\tilde w=\text{acc\_int}_j+\text{acc\_age}_j d+w$. Si se queda la forma de la tesis: el nuevo nace en $w_{0j}=\text{acc\_int}_j$ (por marca, no "fijo" a secas) y el grid debe cubrir más rango. O adoptar la forma del código. |
| Transición de $w$ (l. 64) | $m(j,a,r,w)$ general | $w+\delta_j-\text{acc\_age}_j-\kappa r$; $\delta_j=\text{acc\_age}_j+\kappa\,\bar p$, $\kappa=0.25$, $\bar p=0.3$ | Escribir la forma funcional y la calibración. |
| Notación $\sigma_w$ (l. 64) | sd de $\eta$ | `sigma_eta = 0.15` es la sd; `sigma_w = 0.1` es la escala del nido de elegir $w$ al comprar | Usar $\sigma_\eta$ para el choque y $\sigma_w$ para el nido. |
| Grid de $w$ (l. 37, 54, 62) | $w\in(-\infty,\infty)$ (l. 37 escribe $\{-\infty,\infty\}$) | 71 puntos en $[-3,2.25]$, paso 0.075 ($=\sigma_\eta/2$); extremos absorben colas; piso `eps_F` | Mencionar la discretización y que en las fronteras $E[\eta\mid w]\neq0$. |
| Precio de reparar (l. 48) | "U invertida o logarítmica", bajo de nuevo y creciente | LB: $4+0.1(a-1)+6e^{-(a-1)/4}$ → 10.0 (edad 1), 6.6 (5), 5.5 (12), 6.2 (23): **forma de U** (caro de joven por piezas nuevas). HB: $10+0.6(a-1)$, lineal 10 → 23 | Reescribir la frase y dar la fórmula. |
| Regímenes de $R$ | "régimen de precios de reparación" | $R_t=R\,e^{\zeta_t}$, $\zeta\in\{-0.3,0,0.3\}$ | Escribir; ver sec. 4. |
| Reparar el nuevo (árbol, l. 75-108) | Tras "escoger $j$ luego $a$" se repara, incluidos nuevos | Nuevos no se reparan (nacen en $w=0$); tampoco la edad $\bar a-1$ | Agregar la frase. |
| $w$ en edad $\bar a-1$ (l. 54) | $(j,\bar a-1,\emptyset)$: $w$ irrelevante | El estado sigue indexado por $w$: utilidad tiene `u_w·w` ($u_w=-0.2$), así que $P(j,\bar a-1,w)$ varía con $w$; solo no se repara | Ajustar el texto (con $u_w=0$ sería cierto para la utilidad). |
| Utilidad (l. 41) | $u(j,a,w)$ | $u_{0,\tau j}+u_{1,\tau j}d+u_2 d^2+u_w w$, $u_2=0$; $u_0,u_1$ de Gillingham | Escribir la forma. |
| Nidos (l. 153) | $\sigma$, $\sigma_D$, $\sigma_r$, $\sigma_y$ (marca → edad → $w$) | `sigma = sigma_trade = 1` (fijas), `sigma_sell = 0.3454`, `sigma_rep = 0.3`, `sigma_w = 0.1`; el nido es sobre $(j,d)$ y luego $w$ | Con $\sigma_y=1$ el nido de marca colapsa: equivalente. Decir qué escalas se fijan y renombrar $\sigma_D\to\sigma_{sell}$ si se quiere. |
| Venta/chatarra en el intercambio (árbol) | El árbol solo pone vender/deshuesar en "Deshacerse" | También al intercambiar se elige vender o chatarrear el coche actual (como en la fórmula de $\Pi$, l. 173) | Agregar $D$ a la rama de intercambiar en el árbol. |
| Chatarra $\underline P_j$ | por marca | `p_scrap = 2.0` igual para las dos marcas (calibración propia) | Mencionar. |
| $T_b$ (l. 46) | un $T_b$ | `tc_buy` por tipo y `tc_buy_nocar`, que **reemplaza** a `tc_buy` si no se tenía coche (lectura de la Tabla 10) | Escribir los dos y la lectura. |
| Terminal | — | El dueño de un terminal puede comprar el mismo periodo (`trade_term`) | Mencionar el supuesto. |

## 4. Cómo describir el Monte Carlo

- Cada régimen $t$ es un **mercado distinto en su propio equilibrio estacionario**
  $(EV_t,P_t,q_t)$, con $R_t$ percibido como permanente (regiones o periodos lejanos), no
  años sucesivos para los mismos hogares. $P$ no se recalcula en el tiempo.
- Panel: por régimen, $N=10{,}000$ hogares distintos (mitad por tipo), estado inicial
  sorteado de $q_{\tau,t}$, $K=4$ años con $P_t$ fijo; la distribución agregada sigue
  estacionaria. Contribuyen $N(K-1)\times3\approx90{,}000$ transiciones.
- Estimación: para cada $\theta$ se resuelven los 3 equilibrios; la variación de $R_t$ entre
  mercados mueve $\Pr(r)$ pero no $F_r$ (Hu & Xin, Assumption 7) e identifica la mezcla.
- Tres diseños sobre el mismo panel (1: $r$ y motivo de salida observados; 2: sin $r$;
  3: sin $r$ ni motivo) + estimador de Gillingham ignorando $w$. $R=50$ réplicas.
- Si se quisieran regímenes como años sucesivos haría falta dinámica de transición (choques
  sorpresa) o $R$ como estado markoviano: no está implementado.
