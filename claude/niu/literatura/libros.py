"""Comprueba en Open Library los libros sin DOI y los últimos enlaces; también muestra autores de dos casos dudosos."""
import json
import pathlib
import sys
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
AQUI = pathlib.Path(__file__).parent
H = {"User-Agent": "literatura-tesis/1.0"}
LIBROS = [
    ("Numerical Methods in Economics", "Judd"),
    ("Applied Computational Economics and Finance", "Miranda"),
    ("Dynamic Economics", "Adda"),
    ("Econometric Analysis of Cross Section and Panel Data", "Wooldridge"),
    ("Dynamic Programming and Markov Processes", "Howard"),
    ("The Waste Makers", "Packard"),
    ("Applied Mathematical Programming", "Bradley"),
    ("Frontiers in Econometrics", "Zarembka"),
    ("Spatial Interaction Theory and Planning Models", "Karlqvist"),
]
URLS = [
    "https://ideas.repec.org/a/aea/aecrev/v65y1975i1p148-57.html",
    "https://ideas.repec.org/p/bdm/wpaper/2022-02.html",
    "https://cowles.yale.edu/sites/default/files/2022-12/d0477.pdf",
    "https://openyls.law.yale.edu/bitstream/handle/20.500.13051/8128/10_21YaleJonReg67_2004_.pdf",
    "https://hal.science/hal-04535706",
    "https://www.imf.org/en/Publications/WP/Issues/2020/05/15/Global-Smartphone-Sales-May-Have-Peaked-What-Next-49374",
]

for titulo, autor in LIBROS:
    q = urllib.parse.urlencode({"title": titulo, "author": autor, "limit": 1, "fields": "key,title,author_name,first_publish_year,publisher"})
    try:
        with urllib.request.urlopen(urllib.request.Request("https://openlibrary.org/search.json?" + q, headers=H), timeout=40) as r:
            d = json.load(r)
        doc = (d.get("docs") or [{}])[0]
        print("LIBRO", titulo, "->", doc.get("title"), doc.get("author_name"), doc.get("first_publish_year"),
              "https://openlibrary.org" + doc["key"] if doc.get("key") else "SIN RESULTADO")
    except Exception as e:
        print("LIBRO", titulo, "ERROR", e)

for u in URLS:
    try:
        with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=40) as r:
            t = r.read(4000)
            print(r.status, u, t[:5] if t[:4] == b"%PDF" else t[:160])
    except Exception as e:
        print("ERR", u, e)

m = json.loads((AQUI / "metadatos.json").read_text(encoding="utf-8"))
for k in ["Zurn2025", "RightToRepairRRC2025", "HMMMaintenance2022", "ChowdhuryVerma2024b", "Blundell2026", "Blondel2022", "Fachbach2022"]:
    x = m.get(k)
    print(k, x and (x["autores"], x["anio"], x["revista"], x["vol"], x["pag"], x["doi"], x["titulo"][:70]))
print(sorted(k for k, x in m.items() if x["tipo"] in ("posted-content", "report", "dissertation", "other", "peer-review", "component")))
