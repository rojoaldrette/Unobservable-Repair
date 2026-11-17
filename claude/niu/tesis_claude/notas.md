# Notas sobre `tesis_claude/tesis.tex`

Versión de Claude de `docs/tex/tesis.tex` (2026-10-29). Compila con
`pdflatex tesis.tex` (dos veces); 41 páginas. No se tocó `docs/tex/tesis.tex`.

## 1. Qué cambió respecto a tu tesis

### Correcciones de errores (las que habíamos comentado)

Salen de `niu/docs/diferencias_tesis_codigo.md` y `niu/docs/tesis_transiciones.tex`.

- **Dimensiones.** $P\in\mathbb R^{J(\bar a-1)W}$; $q\in\mathbb R^n$ con
  $n=J(\bar a-1)W+J+1$; simplex de dimensión $n-1$. El terminal y el nuevo son **un solo
  estado** por marca.
- **Demanda.** Ahora separa a los dueños de terminal, que no tienen $w$.
- **Ω.**
  - Los subbloques tienen tamaños distintos ($\Psi$, $\psi$, $\phi$, escalar).
  - Se quitó el 0 que sobraba en $\Xi$.
  - El terminal no se puede mantener.
- **Q.**
  - $1-s$ va en una diagonal $\bar S_{ja}$ y $s$ en la columna terminal.
  - La edad $\bar a-1$ va toda a terminal.
  - Fila del nuevo.
  - Como $s$ ahora depende de $(j,a,w)$, los bloques cambian por marca y edad. Se quitó la
    frase "la diferencia por marca radica en qué $w$ se agrupa cada marca".
- **A.**
  - $C_{ja}$ usa la reparación de la edad $a-1$.
  - $C_{j1}$ es $f(\cdot\mid r=0)$.
  - El terminal es un 1.
  - Se agregó la nota de por qué la factorización $QA$ es válida (opción 2).
- **Verosimilitud.**
  - $Z(w'\mid j,a,w)$ usa $f(\cdot\mid j,a,w,r)$. Tu texto ponía $a+1$, que no cuadraba
    con la definición de $Z$.
  - La celda incluye $w'$.
  - Se agregó el caso del nuevo $\delta_t=(j,0)$.
  - Se explica por qué no se suman las $2^{T-1}$ secuencias de reparación.
- **$s=\text{logit}(w)$** se cambió a $s=\Lambda(\cdot)$, la inversa del logit.
- **$w\in\{-\infty,\infty\}$** se cambió a $(-\infty,\infty)$.
- **Coche de edad $\bar a-1$.** Sí conserva su $w$, porque afecta su utilidad y su precio;
  solo no se repara.
- **Nidos.**
  - La fórmula de $EV$ en "Probabilidades" estaba mal anidada: el nido de vender o
    chatarrizar ($\sigma_D$) también está dentro de *intercambiar*, no solo de
    *deshacerse*.
  - La probabilidad de mantener ahora usa $\exp(v/\sigma)$ con los mismos tres términos en
    el denominador.
  - Se agregaron las condiciones $\sigma_w\le\sigma_y\le\sigma$, etc.
- **Árbol.**
  - Ahora la rama *intercambiar* incluye vender o chatarrizar el coche actual.
  - Se corrigió "on coche".
  - El nodo "Mantener $j=K$" pasó a "$h=(j,a,w)$".
  - El pie de figura dice qué no se repara.
- **Teorema 1.** $A(P)$ pasó a $A(P,R)$, y "satisfacen" a "satisface".
- **Gillingham et al.** Tu texto lo citaba como 2021, 2022 y 2024. Todo quedó como
  **(2022)**: JPE 130(10), 2534-2593.
- **BLP.** "los precios son otro parámetro a estimar" no es exacto. Los precios no son
  parámetros libres: se sustituyen por $P(\theta)$, y no hay instrumentos porque no hay
  características no observadas correlacionadas con el precio.
- **Costos de transacción.**
  - Están en utils, así que el costo total es $\mu(P_{buy}-P_{sell})+T_b+T_s$.
  - Se agregó $T_b^\emptyset$, que reemplaza a $T_b$ si el comprador no tenía coche.
  - La nota al pie ahora dice qué es lo que no tomamos de Gillingham: el costo extra del
    vendedor en años de inspección. Así, $T_s$ queda constante.
- **Ortografía y gramática.** Se corrigieron, entre otras:

  | Antes | Después |
  |---|---|
  | compraar | comprar |
  | desacerse | deshacerse |
  | normalzado | normalizado |
  | acciente | accidente |
  | deshechar | desechar |
  | verosilimitud | verosimilitud |
  | estrategía | estrategia |
  | Gilligham | Gillingham |
  | especifico | específico |
  | termino | término |
  | instantanea | instantánea |
  | distrubución | distribución |
  | mase | masa |
  | $0\ge q\ge1$ | $0\le q\le1$ |
  | $U(\dot)$ | $U(\cdot)$ |

  También se agregaron acentos y comas en varios lugares.

### Especificación que tomé como la tuya

La tomé de `modelo_fin.md`, de la presentación y de los pendientes de la memoria.

- **$s=\Lambda(\alpha_{0j}+\alpha_{1j}a+\alpha_{2j}a^2+w)$**, con $w$ como desviación de
  la curva de Gillingham.
- **Transición.**
  - $w'=w+\delta_j-\gamma_j(a)-\kappa r+\eta$, con $\gamma_j(a)=\alpha_{1j}+\alpha_{2j}(2a+1)$.
  - Es la forma de la presentación.
- **Utilidad.** $u=u_0+u_1a+u_w w$, sin $u_2$: la edad² en la utilidad sigue siendo solo
  idea.
- **Ts constante**, sin inspección.
- **R(j,a)** con prima para jóvenes en LB y lineal en HB; regímenes $R e^{\zeta}$.
- **Quién decide qué.**
  - El dueño de un terminal puede comprar en el mismo periodo.
  - Los nuevos y los de edad $\bar a-1$ no se reparan.

### Cosas que reescribí más allá de la ortografía (revísalas)

- **Precio de reparación.** Tu texto decía "U invertida o logarítmica… bajos de nuevos por
  la garantía". La especificación aprobada es la contraria: la prima hace a la LB *más
  cara* de joven (piezas), y HB es lineal creciente. Lo reescribí con la fórmula. Si
  prefieres la historia de la garantía, hay que cambiar el código, no solo el texto.
- **"No hay límites en cuanto es el precio de reparación".** No entendí la idea, así que la
  reemplacé por "es exógeno y no depende de $w$". Revísalo.
- **Supuestos clave.** La frase "$P\ge$ chatarra si y solo si los costos de transacción no
  son muy altos" quedó suavizada: "si los costos… no son muy altos, normalmente…". El "si y
  solo si" no es cierto en general.
- **Regímenes.** Ahora dicen explícitamente que son mercados distintos en su propio
  equilibrio, no años sucesivos.
- **El párrafo "(ESTO PUEDE IR EN EL APÉNDICE)".** Se convirtió en la subsección
  *Identificación de la reparación no observada*:
  - mapeo a los supuestos 1, 2 y 7 y a la sec. 6.3 de Hu & Xin;
  - advertencia de selección (el $w'$ solo se ve si no se chatarriza);
  - el problema de separar $T_b$ y $T_s$, que es por qué usamos dos tipos con $\mu$
    distinta.
- **Sección "Estimación".** Se renombró a **Resultados**, porque tu instrucción era poner
  ahí el esquema de resultados.

### Desarrollo matemático agregado (pediste más explicación)

- **Funciones de valor por decisión**, completas y en dos etapas:
  - reparación: $v_0$, $v_1$, $\mathcal W(h)$ y $\Pr(r)$;
  - intercambio: $v_K$, $v_\emptyset$ y $v_T$, con los valores inclusivos $\mathcal I_T$ y
    $\mathcal I_{ib}$;
  - terminal y sin coche.

  Esto reemplaza el "Omitimos el desarrollo… se dejará para el desarrollo final".
- **El operador de Bellman.**
  - Condiciones de Blackwell (monotonía y descuento), con la prueba de por qué el log-sum
    las cumple.
  - Contracción de módulo β y **teorema del punto fijo de Banach**: existencia, unicidad y
    convergencia $\beta^k$.
  - Poli-algoritmo: iteraciones sucesivas y luego Newton-Kantorovich, con el jacobiano
    $\beta\mathcal M$ por el teorema de Williams-Daly-Zachary.
- **Distribución estacionaria.** Existencia y unicidad: como todas las CCP son positivas,
  hay una sola clase recurrente que contiene a $\emptyset$, y es aperiódica.
- **Existencia del equilibrio $(P,q)$.**
  - No la probamos. Se cita que en modelos parecidos se prueba con argumentos de punto
    fijo (Rust 1985; Konishi & Sandfort 2002). Esos argumentos son tipo **Brouwer**, que
    es distinto de Banach.
  - Se dice que se verifica numéricamente y se resuelve con Newton conjunto.
  - Ojo: en tu mensaje decías "Banach Brouwer". Son dos teoremas distintos: Banach se usa
    para Bellman (contracción) y Brouwer o Kakutani para la existencia del equilibrio de
    precios. El texto los separa así.
- **Discretización.** Tauchen, con la advertencia de $E[\eta\mid w]\neq0$ en las fronteras.
- **Estimador NFXP.** Gradiente por el teorema de la función implícita.

## 2. Pendientes que tienes que decidir

1. **Valor de $\alpha_{2j}$.** Gillingham no tiene edad² en accidentes. En la tabla del MC
   quedó "[por definir]".
2. **$\delta_j$ con edad².**
   - El texto dice que $\delta_j$ se calibra para que, con $\bar p=0.3$, el desgaste
     promedio reproduzca la curva de Gillingham.
   - Con $\alpha_{2j}\neq0$ y $\delta_j$ constante, eso solo se cumple en promedio entre
     edades: en $w$ hay deriva $\delta_j-\gamma_j(a)-\kappa\bar p$, que cambia con la edad.
   - La alternativa es $\delta_j(a)=\gamma_j(a)+\kappa\bar p$, que deja la deriva de $w$ en
     $\kappa(\bar p-r)$.
   - Hay que decidirlo junto con el cambio en el código.
3. **$T_s$.** Puse 0.9106 (el $T_s$ de Gillingham en años sin inspección). Si al quitar la
   inspección lo quieres recalibrar, cambia la tabla.
4. **Número de parámetros.** Escribí 28: los 27 del MC anterior, menos `tc_sell_inspect`,
   más $\alpha_{2j}$ por marca. Hay que confirmarlo con el código final.
5. **$W=71$ puntos en $[-3, 2.25]$.** Hay que confirmar que el rango sigue sirviendo con la
   edad² (diagnóstico de masa en los extremos < 0.5%).
6. **Introducción.** Solo dejé un esquema por párrafos, hecho con tus notas personales. No
   la pediste explícitamente.
7. **Conclusiones.** No hay sección; se escribe con los resultados.

## 3. Literatura: qué metí y qué no

Todas las citas del texto las verifiqué en los PDF de `po/Fuentes`, en
`docs/papers` (Gillingham, Hu & Xin y sus listas de referencias) o con búsqueda web.

### Hay que revisar páginas y números de estas

Los autores, el año y la revista los tengo seguros. Las páginas o el número salen de
memoria.

| Referencia | Qué revisar |
|---|---|
| Strausz | Economic Journal 2009, 119(540). En `po` tienes el WP de 2006. |
| Orbach (2004) | Páginas: Yale J. on Regulation 21(1), 67-118. |
| Parajuly et al. | J. Industrial Ecology: el año 2024 y las páginas (DOI 10.1111/jiec.13451). |
| Copeland & Monnet (2009) | Páginas (RES 76(1)). |
| Misra & Nair (2011) | Páginas. |
| McGrattan & Schmitz (1999) | La búsqueda no lo encontró. Estoy casi seguro de que es FRB Minneapolis Quarterly Review 23(4). |
| Moral & Jaumandreu | Lo puse como 2007, Spanish Economic Review 9(3). Tu protocolo decía 2006, que es la fecha en línea. |
| Kennet (1994) | Número 9(4). |

### Metido, por sección

- **Durables:** Coase 1972; Gul, Sonnenschein & Wilson 1986; Bulow 1982, 1986 (el "bulow
  seminal" de `po`); Swan 1970, 1972; Waldman 1993, 1996, 2003; Fishman et al. 1993; Lee
  & Lee 1998; Grout & Park 2005; Strausz; Orbach 2004; Fisher, Griliches & Kaysen 1962;
  Fukasawa 2025 (sale del reporte de Elicit).
- **Mercado secundario:** Akerlof 1970; Rust 1985, 1986; Anderson & Ginsburgh 1994;
  Hendel & Lizzeri 1999a, b; Konishi & Sandfort 2002; Stolyarov 2002; House & Leahy 2004;
  Manski 1983; Berkovec 1985; Esteban & Shum 2007; Chen, Esteban & Shum 2013; Schiraldi
  2011; Gavazza, Lizzeri & Roketskiy 2014; Biglaiser et al. 2020; Ghose 2009; Ishihara &
  Ching 2019.
- **Reparación:**
  - Drinkwater & Hastings 1967 (de `po`), Rust 1987, Kennet 1994 y Manski & Goldin 1983.
  - Hamilton & Macauley 1999. De `po` es el WP "Competition and car longevity"; cito la
    versión publicada en el JIE.
  - McGrattan & Schmitz 1999, Hubbard 1998, Jin, Yang & Zhu 2023, Bakker et al. 2014 y
    Parajuly et al.
- **Elección discreta:** McFadden 1973, 1978, 1981; Berry 1994; BLP 1995; Moral &
  Jaumandreu; Rust 1987, 1994; Hotz & Miller 1993; Aguirregabiria & Mira 2002, 2010;
  Keane & Wolpin 1997; Su & Judd 2012; Iskhakov et al. 2016; Mannering & Winston 1985;
  Adda & Cooper 2000; Gowrisankaran & Rysman 2012; Ericson & Pakes 1995; Goettler &
  Gordon 2011; Gillingham et al. 2022; Bellman 1957; Blackwell 1965; Stokey, Lucas &
  Prescott 1989.
- **No observables:** Magnac & Thesmar 2002; Arcidiacono & Miller 2011, 2020; Kasahara &
  Shimotsu 2009; Hu & Shum 2012; Rabiner 1989 (de `po`); Misra & Nair 2011; Copeland &
  Monnet 2009; Gayle & Miller 2015; Perrigne & Vuong 2011; Hu & Xin 2024.
- **Métodos del MC:** Tauchen 1986; BHHH 1974; Liu & Nocedal 1989; White 1982 (valores
  pseudo-verdaderos para interpretar el sesgo de Gillingham).

### Sugerencia para tu punto 6 (acéptala o recházala)

Escribí la sección **2.6 "Vida útil de los vehículos, chatarrización y política pública"**
(Adda & Cooper 2000; Jacobsen & van Benthem 2015; Bento, Roth & Zuo 2018; Gillingham;
Bakker et al. 2014). Conecta con tu motivación ambiental y con los contrafactuales de
*right to repair*. Tiene un comentario `% Subsección sugerida`: si no la quieres, se borra
sin romper nada. Otras opciones que **no** escribí:

- **Información asimétrica en usados.** Una subsección propia sobre lemons: Akerlof,
  Hendel & Lizzeri, House & Leahy, Biglaiser et al. y Hubbard. Hoy está repartida en 2.2 y
  2.3. Vale la pena si en algún momento quitas la información simétrica (era el plan B de
  `modelo_fin.md`).
- **Aftermarkets y antitrust.** El caso Kodak y la literatura de mercados de refacciones y
  servicio cautivos. Es muy relevante para *right to repair*, pero lo veo lejos de tu
  modelo, que no tiene oferta.
- **Métodos computacionales para equilibrios.** Rust, Traub & Woźniakowski 2002; Ma &
  Stachurski 2021, ambos citados en Gillingham. Solo los pondría si en la tesis defiendes
  el algoritmo como contribución.

### Lo que encontré "parecido a lo tuyo"

No encontré ningún trabajo que estime reparación no observada en un modelo de mercado.
Lo más cercano:

1. **Hu & Xin (2024).** Re-estiman Rust (1987) sin observar los reemplazos. Es tu
   estrategia aplicada a mantenimiento, pero sin mercado ni precios.
2. **Kennet (1994).** Mantenimiento de motores de avión, pero la decisión se observa.
3. **Gillingham et al. (2022).** Interpretan $\epsilon$ como costos de mantenimiento no
   observados, y el zig-zag como "chatarrizar en vez de reparar para la inspección". Es el
   mejor argumento de motivación: el mismo paper reconoce que la reparación está ahí,
   escondida.
4. **Manski & Goldin (1983).** La chatarrización depende del precio neto de reparación.

El texto dice "hasta donde sabemos" en ambos lugares.

### Lo que NO metí de `po`, y por qué

- **Fuera de tema:**
  - Gupta & Romano 1998 (monitoreo principal-agente);
  - Atalay et al. 2025 (productividad manufacturera);
  - Dubé, Hitsch & Chintagunta 2010 (efectos de red);
  - Hashmi & Van Biesebroeck 2016 (innovación);
  - Neuts et al. 2000 (teoría de colas de reparación, demasiado técnica);
  - Leeson (bienestar);
  - "Logic is a harsh mistress".
- **Sin rigor suficiente o fuera de economía:**
  - de Oliveira 2013;
  - Kuppelwieser et al. (respuestas del consumidor a la obsolescencia programada);
  - Ates & Acur 2022;
  - Schreiber et al. (publicidad);
  - Oraee et al. 2024 (smartphones);
  - Satyro et al. 2018;
  - "Beyond planned obsolescence" (GAIA);
  - Guiltinan 2009 (ética);
  - Huisman (e-waste);
  - Slade, *Made to Break* (libro de historia).

  Algunos podrían ir en la introducción como motivación, no en la revisión.
- **Dudé y lo dejé fuera:**
  - Kinokuni, Ohori & Tomoda 2021 (cuotas de disposición y durabilidad del monopolista).
    Es buena teoría, pero el modelo no tiene esa política.
  - Oguchi & Fuse 2015 (distribución de vida útil de autos). Podría ir en 2.6 como
    evidencia de medición de vida útil.
  - Sabbaghi et al. (encuesta de reparabilidad). No pude verificar la revista.
  - Fernández 1998 (modelo de duración del reemplazo de durables, Cuadernos de Economía).
    Es en español y estructural; podría ir en 2.4 si quieres una referencia
    latinoamericana.
  - Adda & Cooper (2006), "Dynamics of car sales". Cité su artículo de 2000, que es el de
    chatarrización. Tu protocolo citaba el WP "2008".
  - El libro *Durable goods price estimation* (Gordon) y *The Performance Economy*
    (Stahel). Útiles para medición y motivación, no para la revisión.
- **Datos** (INEGI, Profeco, IDC). No aplican al MC.

## 4. Fuentes nuevas (no estaban en `po` ni en `docs/papers`)

| Referencia | De dónde | Para qué |
|---|---|---|
| Rust (1985), Econometrica 53(4) | lista de Gillingham | primer equilibrio estacionario |
| Rust (1986), Econometrica 54(1) | lista de Gillingham | matar el mercado de usados |
| Berkovec (1985); Manski (1983); Manski & Goldin (1983) | lista de Gillingham | equilibrio estático y chatarrización con costos de reparación |
| Konishi & Sandfort (2002); Stolyarov (2002); House & Leahy (2004) | lista de Gillingham | teoría de equilibrio con usados |
| Hendel & Lizzeri (1999a, b); Akerlof (1970) | lista de Gillingham / conocimiento | selección adversa |
| Biglaiser et al. (2020); Mannering & Winston (1985); Adda & Cooper (2000) | lista de Gillingham | usados y reemplazo |
| Hamilton & Macauley (1999), JIE 47(3) | búsqueda web | longevidad y precios de refacciones |
| Kennet (1994), JAE | búsqueda web | mantenimiento estructural |
| Hubbard (1998), RAND 29(2) | búsqueda web | mercado de inspección vehicular |
| Bento, Roth & Zuo (2018), Energy Journal 39(1) | búsqueda web | vida útil en EE. UU. |
| Jacobsen & van Benthem (2015), AER | conocimiento | chatarrización y política |
| McGrattan & Schmitz (1999) | conocimiento (verificar) | gasto agregado en reparación |
| Magnac & Thesmar; Kasahara & Shimotsu; Hu & Shum; Arcidiacono & Miller 2020; Misra & Nair; Copeland & Monnet; Gayle & Miller; Perrigne & Vuong | lista de Hu & Xin | identificación |
| Blackwell (1965); Stokey, Lucas & Prescott (1989); Bellman (1957); Tauchen (1986); White (1982); Liu & Nocedal (1989) | conocimiento | matemáticas y métodos |
| Swan (1970) AER; Bulow (1986) QJE | conocimiento | durabilidad (el PDF "bulow seminal" de `po` no tiene texto extraíble; supuse que es Bulow 1986) |

## Revisión de literatura reestructurada (2026-11-08)

- Seis subsecciones orientadas a la pregunta sobre la importancia de la reparación, con transiciones:
  durables, mercado secundario, reparación, modelos dinámicos, equilibrio de coches, identificación.
- Se quitó la subsección de vida útil y política: lo esencial pasó a la subsección de reparación y al
  esquema de la introducción.
- Correcciones: Rust (1985) y Stolyarov (2002) se ajustaron a lo que dicen sus resúmenes publicados.
- Diez referencias nuevas. `niu/utils_tex/bib/referencias.bib` todavía no las incluye.
