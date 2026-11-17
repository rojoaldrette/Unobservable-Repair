"""Nido 5: equilibrio estructural del mercado de autos."""
from base import F, T

SEC5 = ("Equilibrio estructural del mercado de autos",
 "De aquí sale tu modelo. Los precios de los usados son endógenos y vacían el mercado, y las decisiones de los hogares se agregan en una distribución de coches. La sección cierra diciendo qué decisión les falta a estos modelos.",
 [("Lo que tienes: modelos de equilibrio", [
   T("Gillingham2022", "Aportación: modelo tratable de equilibrio estacionario con consumidores heterogéneos y un algoritmo rápido y robusto; concluye que Dinamarca podría recaudar más reduciendo el impuesto al registro. Método: estimación con casi 39 millones de transiciones de propiedad en Dinamarca. Contexto: es el modelo que extiendes y le falta precisamente una decisión explícita de reparar. Tienes también el NBER 25840.", "Gillingham Rust et al TradeAutos (2019).pdf; rust2019.pdf"),
   T("Schiraldi2011", "Aportación: procedimiento explícito para estimar costos de transacción con nuevos y usados, a partir de quienes conservan y quienes compran un tipo de auto. Método: coeficientes aleatorios con paro óptimo dinámico, aplicado a subsidios de chatarreo en Italia. Contexto: tu nota de identificación de costos de transacción dialoga con esta estrategia.", "The RAND J of Economics - 2011 - Schiraldi.pdf"),
   T("GavazzaLizzeriRoketskiy2014", "Aportación: los costos de transacción afectan mucho el volumen, la asignación y el mercado primario, aunque el efecto agregado en bienestar es relativamente pequeño. Método: modelo calibrado a EE. UU. y Francia con heterogeneidad en la disposición a pagar. Contexto: es el referente para preguntar cuánto importa una fricción en equilibrio, igual que tú con la reparación. Tienes dos versiones.", "Gavazza-QuantitativeAnalysisUsedCar-2014.pdf"),
   T("EstebanShum2007", "Aportación: mide la importancia competitiva del mercado secundario para los fabricantes. Método: oligopolio dinámico estimado con 20 años de datos de autos. Contexto: tiene la oferta endógena que tu modelo deja fuera. Tienes dos versiones.", "Esteban-DurableGoodsOligopolySecondary-2007.pdf"),
   T("ChenEstebanShum2013", "Aportación: abrir el mercado secundario reduce 35\\% las ganancias de fabricantes en su calibración, menos si cae el acervo de usados, por ejemplo si los productos son menos durables. Método: oligopolio dinámico durable con costos de transacción. Contexto: conecta durabilidad y equilibrio, y sugiere que tu reparación también importaría para la oferta.", "chen esteban 2013.pdf"),
  ]),
  ("Lo que tienes: oferta dinámica", [
   T("EricsonPakes1995", "Aportación: marco de industria con entrada, salida e incertidumbre específica por empresa. Método: equilibrio de Markov perfecto con expectativas racionales y ergodicidad. Contexto: es la base si algún día endogeneizas la oferta; en tu modelo actual la oferta es exógena.", "ericson & Pakes 1995 Markov.pdf"),
   T("GoettlerGordon2011", "Aportación: la competencia de AMD reduce la tasa de innovación 4.2\\%, pero sin AMD cae el excedente. Método: oligopolio dinámico con durables e innovación endógena, estimado. Contexto: muestra cómo se estima oferta dinámica con durables.", "GoettlerGordon2011.pdf"),
   T("DubeHC2010", "Aportación: mide el tipping por efectos de red indirectos. Método: demanda y equilibrio dinámico en consolas de videojuegos. Contexto: es metodológicamente útil y lejano en el tema; cítalo poco.", "Dub-TippingConcentrationMarkets-2010.pdf"),
   T("HashmiVB2016", "Aportación: relación de equilibrio entre estructura de mercado e innovación en autos. Método: juego dinámico estimado, con patentes como proxy. Contexto: oferta dinámica en tu misma industria.", "Hashmi-RELATIONSHIPMARKETSTRUCTURE-2016.pdf"),
  ]),
  ("Lo que falta: equilibrio de autos", [
   F("Manski1983", "Aportación: análisis de equilibrio de la tenencia de autos en Israel. Método: elección discreta agregada. Contexto: de los primeros equilibrios numéricos de este mercado.", True),
   F("Berkovec1985", "Aportación: equilibrio general de corto plazo que combina demanda de elección discreta, producción de nuevos y chatarreo. Método: estimación y simulación de 1978 a 1990. Contexto: pionero con heterogeneidad amplia y antecedente directo de Gillingham et al.", True),
   F("Blundell2026", "Aportación: estima que la penalización por lemons es 12\\% del precio en el primer año de tenencia y cae después. Método: equilibrio de tenencia de autos con información privada y registros de Dinamarca. Contexto: es el trabajo más cercano en espíritu: toma el mismo tipo de modelo y agrega algo no observado sobre la calidad, pero no modela la reparación.", True),
   F("Gillingham2018", "Aportación: versión con uso del vehículo y tipo de auto. Método: modelo dinámico de tenencia, tipo y uso. Contexto: sirve para ver lo que quedó fuera de la versión de 2022."),
   F("Bento2009", "Aportación: impactos de impuestos a la gasolina en un modelo que une nuevos, usados y chatarra con hogares y autos heterogéneos. Método: estimación conjunta de tenencia y kilómetros. Contexto: una alternativa de equilibrio orientada a política."),
  ]),
  ("Lo que falta: equilibrio estacionario y distribuciones", [
   F("LucasPrescott1971", "Aportación: equilibrio recursivo con expectativas racionales. Método: teoría. Contexto: es el origen de la técnica."),
   F("Hopenhayn1992", "Aportación: equilibrio estacionario con entrada, salida y heterogeneidad. Método: teoría. Contexto: es la analogía natural de empresas que nacen y mueren con tus coches."),
   F("HopenhaynPrescott1992", "Aportación: existencia y estabilidad de distribuciones invariantes en procesos monótonos. Método: punto fijo de Knaster-Tarski y supermodularidad. Contexto: útil para hablar de tu distribución estacionaria q.", True),
   F("Huggett1993", "Aportación: equilibrio estacionario con agentes heterogéneos, donde un precio vacía el mercado dada una distribución. Método: cálculo numérico. Contexto: misma estructura de solución que la tuya, en otro mercado."),
   F("Aiyagari1994", "Aportación: modelo de crecimiento con ahorro precautorio y restricciones de liquidez. Método: análisis cualitativo y cuantitativo. Contexto: referente clásico de la distribución estacionaria de agentes."),
   F("Young2010", "Aportación: simula la distribución sin Monte Carlo mediante un histograma. Método: método numérico. Contexto: es la misma idea de tu matriz de transición.", True),
  ]),
  ("Lo que falta: oligopolio dinámico", [
   F("PakesMcGuire1994", "Aportación: algoritmo para calcular equilibrios de Markov perfectos con productos diferenciados. Método: numérico. Contexto: es el punto de partida para calcular este tipo de equilibrios."),
   F("PakesMcGuire2001", "Aportación: algoritmo estocástico que no está atado a la maldición de dimensionalidad. Método: aproximación con promedios de resultados pasados. Contexto: sería relevante con oferta endógena."),
   F("DoraszelskiPakes2007", "Aportación: marco para aplicar modelos dinámicos de organización industrial. Método: revisión. Contexto: es el manual de referencia."),
   F("BBL2007", "Aportación: estima juegos dinámicos en dos etapas con controles discretos y continuos. Método: distancia mínima simulada. Contexto: es la opción si estimas oferta sin resolver el juego.", True),
   F("POB2007", "Aportación: estima costos de entrada y salida con promedios muestrales de valores. Método: dos etapas no paramétricas. Contexto: alternativa simple al anterior."),
   F("PesendorferSD2008", "Aportación: marco de mínimos cuadrados asintóticos que unifica a Hotz y Miller y Aguirregabiria y Mira. Método: teoría de estimación y Monte Carlo. Contexto: ayuda a ver en qué difieren esos estimadores."),
   F("AguirregabiriaMira2007", "Aportación: estimación pseudo-máximo-verosímil de juegos dinámicos, atendiendo equilibrios múltiples. Método: secuencial (NPL). Contexto: extiende a juegos el método del nido 6."),
   F("WeintraubBVR2008", "Aportación: concepto de equilibrio simplificado para industrias con muchas empresas. Método: equilibrio oblivious. Contexto: útil si piensas en muchas marcas."),
  ])],
 "Transición a la estimación. Estos modelos resuelven el equilibrio y sus autores lo llevan a datos observando las transiciones de los coches, pero no observan todo lo que hacen los hogares. Si la reparación ocurre en talleres sin registro, no basta con agregarla al modelo: hay que decir cómo se identifica sin observarla, lo que lleva al último nido.")
