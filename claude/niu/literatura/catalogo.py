"""Datos manuales del catálogo y orden de los nidos. Los resúmenes están en sec*.py; los datos bibliográficos, en metadatos.json o MANUAL."""


OL = "https://openlibrary.org/works/"
MANUAL = {
    "Bellman1957": ("Bellman, R. (1957). \\textit{Dynamic Programming}. Princeton University Press (reimpr. 2010).", "https://doi.org/10.1515/9781400835386", "Crossref"),
    "Blondel2022": ("Blondel, M., Q. Berthet, M. Cuturi, R. Frostig, S. Hoyer, F. Llinares-López, F. Pedregosa y J.-P. Vert (2022). ``Efficient and Modular Implicit Differentiation''. \\textit{Advances in Neural Information Processing Systems} 35, 5230--5242.", "https://arxiv.org/abs/2105.15183", "arXiv"),
    "McFadden1973": ("McFadden, D. (1973). ``Conditional Logit Analysis of Qualitative Choice Behavior''. En P. Zarembka (ed.), \\textit{Frontiers in Econometrics}, 105--142. Academic Press.", "https://eml.berkeley.edu/reprints/mcfadden/zarembka.pdf", "PDF"),
    "McFadden1978": ("McFadden, D. (1978). ``Modelling the Choice of Residential Location''. En A. Karlqvist et al. (eds.), \\textit{Spatial Interaction Theory and Planning Models}, 75--96. North-Holland.", OL + "OL19141696W", "volumen en Open Library"),
    "McFadden1981": ("McFadden, D. (1981). ``Econometric Models of Probabilistic Choice''. En C. Manski y D. McFadden (eds.), \\textit{Structural Analysis of Discrete Data}, 198--272. MIT Press.", "https://eml.berkeley.edu/~mcfadden/discrete.html", "libro en línea"),
    "AddaCooper2003": ("Adda, J. y R. Cooper (2003). \\textit{Dynamic Economics: Quantitative Methods and Applications}. MIT Press.", OL + "OL21833189W", "Open Library"),
    "AddaCooperWP": ("Adda, J. y R. Cooper (2000). ``The Dynamics of Car Sales: A Discrete Choice Approach''. NBER Working Paper 7785.", "https://www.nber.org/papers/w7785", "NBER"),
    "Swan1970": ("Swan, P. L. (1970). ``Durability of Consumption Goods''. \\textit{American Economic Review} 60(5), 884--894.", "https://ideas.repec.org/a/aea/aecrev/v60y1970i5p884-94.html", "RePEc"),
    "LevhariSrinivasan1969": ("Levhari, D. y T. N. Srinivasan (1969). ``Durability of Consumption Goods: Competition versus Monopoly''. \\textit{American Economic Review} 59(1), 102--107.", "https://ideas.repec.org/a/aea/aecrev/v59y1969i1p102-07.html", "RePEc"),
    "Liebowitz1982": ("Liebowitz, S. J. (1982). ``Durability, Market Structure, and New-Used Goods Models''. \\textit{American Economic Review} 72(4), 816--824.", "https://ideas.repec.org/a/aea/aecrev/v72y1982i4p816-24.html", "RePEc"),
    "Bond1982": ("Bond, E. W. (1982). ``A Direct Test of the `Lemons' Model: The Market for Used Pickup Trucks''. \\textit{American Economic Review} 72(4), 836--840.", "https://ideas.repec.org/a/aea/aecrev/v72y1982i4p836-40.html", "RePEc"),
    "Su1975": ("Su, T. T. (1975). ``Durability of Consumption Goods Reconsidered''. \\textit{American Economic Review} 65(1), 148--157.", "https://ideas.repec.org/a/aea/aecrev/v65y1975i1p148-57.html", "RePEc"),
    "Orbach2004": ("Orbach, B. Y. (2004). ``The Durapolist Puzzle: Monopoly Power in Durable-Goods Markets''. \\textit{Yale Journal on Regulation} 21(1), 67--118.", "https://openyls.law.yale.edu/handle/20.500.13051/8128", "PDF en tu carpeta; título del enlace no comprobable automáticamente"),
    "Fernandez1998": ("Fernández, V. (1998). ``Un modelo de duración estructural para el reemplazo de bienes durables''. \\textit{Cuadernos de Economía} 35(105), 237--254.", "https://www.jstor.org/stable/23830511", "portada del PDF en tu carpeta"),
    "Judd1998": ("Judd, K. L. (1998). \\textit{Numerical Methods in Economics}. MIT Press.", OL + "OL1856113W", "Open Library"),
    "MirandaFackler2002": ("Miranda, M. J. y P. L. Fackler (2002). \\textit{Applied Computational Economics and Finance}. MIT Press.", OL + "OL16939199W", "Open Library"),
    "Baydin2018": ("Baydin, A. G., B. A. Pearlmutter, A. A. Radul y J. M. Siskind (2018). ``Automatic Differentiation in Machine Learning: A Survey''. \\textit{Journal of Machine Learning Research} 18(153), 1--43.", "https://www.jmlr.org/papers/v18/17-468.html", "JMLR"),
    "BHHH1974": ("Berndt, E., B. Hall, R. Hall y J. Hausman (1974). ``Estimation and Inference in Nonlinear Structural Models''. \\textit{Annals of Economic and Social Measurement} 3(4), 653--665.", "https://www.nber.org/chapters/c10206", "NBER"),
    "Wooldridge2010": ("Wooldridge, J. M. (2010). \\textit{Econometric Analysis of Cross Section and Panel Data}, 2a. ed. MIT Press.", OL + "OL5962178W", "Open Library"),
    "Griliches1961": ("Griliches, Z. (1961). ``Hedonic Price Indexes for Automobiles: An Econometric Analysis of Quality Change''. En \\textit{The Price Statistics of the Federal Government}, 173--196. NBER.", "https://www.nber.org/chapters/c6492", "NBER"),
    "OhtaGriliches1976": ("Ohta, M. y Z. Griliches (1976). ``Automobile Prices Revisited: Extensions of the Hedonic Hypothesis''. En N. Terleckyj (ed.), \\textit{Household Production and Consumption}, 325--398. NBER.", "https://doi.org/10.4337/9781035351442.00029", "Crossref (reimpresión)"),
    "Fukasawa2025": ("Fukasawa, T. (2025). ``When Do Firms Sell High Durability Products? The Case of Light Bulb Industry''. Documento de trabajo, arXiv:2503.23792.", "https://arxiv.org/abs/2503.23792", "arXiv"),
    "GroutPark2005": ("Grout, P. A. e I.-U. Park (2005). ``Competitive Planned Obsolescence''. \\textit{RAND Journal of Economics} 36(3), 596--612.", "https://www.jstor.org/stable/4135231", "portada del PDF en tu carpeta"),
    "McGrattanSchmitz1999": ("McGrattan, E. R. y J. A. Schmitz (1999). ``Maintenance and Repair: Too Big to Ignore''. \\textit{Federal Reserve Bank of Minneapolis Quarterly Review} 23(4), 2--13.", "https://www.minneapolisfed.org/research/quarterly-review/maintenance-and-repair-too-big-to-ignore", "Fed de Minneapolis"),
    "Howard1960": ("Howard, R. A. (1960). \\textit{Dynamic Programming and Markov Processes}. MIT Press.", OL + "OL31511043W", "Open Library"),
    "Packard1960": ("Packard, V. (1960). \\textit{The Waste Makers}. David McKay.", OL + "OL3533448W", "Open Library"),
    "Bradley1977": ("Bradley, S. P., A. C. Hax y T. L. Magnanti (1977). ``Dynamic Programming''. Cap. 11 de \\textit{Applied Mathematical Programming}. Addison-Wesley.", "https://web.mit.edu/15.053/www/AMP-Chapter-11.pdf", "MIT"),
    "Banxico2022": ("Chen, L., B. C. Garcia-Medina y R. Wan (2022). ``Trade Liberalization, Consumption Shifting and Pollution: Evidence from Mexico's Used Vehicle Imports''. Documento de trabajo 2022-02, Banco de México.", "https://ideas.repec.org/p/bdm/wpaper/2022-02.html", "RePEc"),
    "Mongardini2020": ("Mongardini, J. y A. Radzikowski (2020). ``Global Smartphone Sales May Have Peaked: What Next?''. IMF Working Paper 20/70.", "https://doi.org/10.5089/9781513545851.001", "DOI en la portada de tu PDF"),
    "Atalay2025": ("Atalay, E., A. Hortaçsu, N. Kimmel y C. Syverson (2025). ``Why Is Manufacturing Productivity Growth So Low?''. NBER Working Paper 34264.", "https://www.nber.org/papers/w34264", "NBER"),
    "HamiltonMacauleyDP": ("Hamilton, B. W. y M. K. Macauley (1998). ``Competition and Car Longevity''. Discussion Paper 98-20, Resources for the Future.", "https://www.rff.org/publications/working-papers/competition-and-car-longevity/", "RFF"),
    "Gillingham2018": ("Gillingham, K., F. Iskhakov, A. Munk-Nielsen, J. Rust y B. Schjerning (2018). ``A Dynamic Model of Vehicle Ownership, Type Choice, and Usage''. Borrador, conferencia AEA 2018.", "https://www.aeaweb.org/conference/2018/preliminary/paper/B6dbT2kD", "AEA"),
    "EUDirective2024": ("Unión Europea (2024). Directiva (UE) 2024/1799 relativa a normas comunes para promover la reparación de bienes.", "https://eur-lex.europa.eu/eli/dir/2024/1799/oj", "EUR-Lex"),
    "EUESPR2024": ("Unión Europea (2024). Reglamento (UE) 2024/1781 sobre requisitos de diseño ecológico para productos sostenibles.", "https://eur-lex.europa.eu/eli/reg/2024/1781/oj", "EUR-Lex"),
    "GEM2020": ("Forti, V., C. P. Baldé, R. Kuehr y G. Bel (2020). \\textit{The Global E-waste Monitor 2020}. UNU/UNITAR/ITU/ISWA.", "https://ewastemonitor.info/gem-2020/", "sitio oficial"),
    "Huisman2012": ("Huisman, J., M. van der Maesen, R. J. J. Eijsbouts, F. Wang, C. P. Baldé y C. A. Wielenga (2012). \\textit{The Dutch WEEE Flows}. United Nations University.", "", "cita tomada de la portada de tu PDF"),
    "Balde2020": ("Baldé, C. P., S. van den Brink, V. Forti, A. van der Schalk y F. Hopstaken (2020). \\textit{The Dutch WEEE Flows 2020: What Happened between 2010 and 2018}. UNU/UNITAR.", "", "cita tomada de la portada de tu PDF"),
    "Dupre2016": ("Dupré, M., M. Jahnich, V. Ramirez, G. Boulbry y É. Ferreira (2016). \\textit{The Influence of Lifespan Labelling on Consumers}. Comité Económico y Social Europeo.", "https://hal.science/hal-04535706", "portada de tu PDF"),
    "JAX": ("Bradbury, J., R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin et al. (2018). \\textit{JAX: Composable Transformations of Python+NumPy Programs}. Software.", "https://github.com/jax-ml/jax", "GitHub"),
    "NotasRust": ("Notas de curso sin autor identificado: ``Lecture Notes: Single-Agent Dynamics''.", "", "solo en tu carpeta"),
    "Elicit": ("Elicit (s.f.). ``Repairability Decisions in Dynamic Oligopoly -- Report''.", "", "solo en tu carpeta"),
}
for _c, _lit, _cap, _pp, _u in [
    ("GordonCh1", "Introduction and Summary of Findings", 1, "3--39", "c8309"),
    ("GordonCh2", "Conceptual Issues in the Measurement of Price and Quality Changes", 2, "40--78", "c8310"),
    ("GordonCh3", "The Methodology of Quality Adjustment", 3, "79--110", "c8311"),
    ("GordonCh8", "New and Used Automobiles", 8, "321--382", "c8316"),
]:
    MANUAL[_c] = (f"Gordon, R. J. (1990). ``{_lit}''. Cap. {_cap} de \\textit{{The Measurement of Durable Goods Prices}}, {_pp}. University of Chicago Press.", f"http://www.nber.org/chapters/{_u}", "NBER")

AUTORES = {
    "SuJudd2012": "Su, C.-L. y K. L. Judd", "DubeFoxSu2012": "Dubé, J.-P., J. T. Fox y C.-L. Su",
    "ArcidiaconoMiller2011": "Arcidiacono, P. y R. A. Miller", "Norets2009": "Norets, A.",
    "KasaharaShimotsu2012": "Kasahara, H. y K. Shimotsu", "ImaiJainChing2009": "Imai, S., N. Jain y A. Ching",
    "KasaharaShimotsu2009": "Kasahara, H. y K. Shimotsu", "CunhaHeckmanSchennach2010": "Cunha, F., J. J. Heckman y S. M. Schennach",
    "PerrigneVuong2011": "Perrigne, I. y Q. Vuong", "WeintraubBVR2008": "Weintraub, G. Y., C. L. Benkard y B. Van Roy",
    "Rust1988": "Rust, J", "Cooper2010": "Cooper, T. (ed.)",
}
ANIO = {"Stewart1994": 1994, "OrtegaRheinboldt1970": 1970, "Cooper2010": 2010, "Slade2006": 2006}

from sec1_durables import SEC1  # noqa: E402
from sec2_secundario import SEC2  # noqa: E402
from sec3_reparacion import SEC3  # noqa: E402
from sec4_dinamicos import SEC4  # noqa: E402
from sec5_equilibrio import SEC5  # noqa: E402
from sec6_identificacion import SEC6  # noqa: E402
from sec7_apoyo import AMBIENTE, MEDICION, NUMERICOS, PERIFERICOS, POLITICA  # noqa: E402

REVISION = [SEC1, SEC2, SEC3, SEC4, SEC5, SEC6]
APOYO = [NUMERICOS, POLITICA, AMBIENTE, MEDICION, PERIFERICOS]
SECCIONES = REVISION + APOYO
