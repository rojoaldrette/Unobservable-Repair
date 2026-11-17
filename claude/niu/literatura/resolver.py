"""Resuelve los DOI fallidos buscando por título y revista, audita el resto y contrasta la bibliografía de la tesis."""
import json
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import metadatos as md  # noqa: E402
import verificar as v  # noqa: E402

AQUI = pathlib.Path(__file__).parent
BIB = AQUI.parent / "utils_tex" / "bib" / "referencias.bib"
PENDIENTES = {
    "Schiraldi2011": ("Schiraldi", "Automobile Replacement: A Dynamic Structural Approach", "RAND"),
    "HendelNevo2006": ("Hendel", "Measuring the Implications of Sales and Consumer Inventory Behavior", "Econometrica"),
    "HouseLeahy2004": ("House", "An sS Model with Adverse Selection", "Political Economy"),
    "KeaneWolpin1997": ("Keane", "The Career Decisions of Young Men", "Political Economy"),
    "HuShumTanXiao2017": ("Hu", "A Simple Estimator for Dynamic Models with Serially Correlated Unobservables", "Econometric Methods"),
    "AguirregabiriaNevo2013": ("Aguirregabiria", "Recent Developments in Empirical IO: Dynamic Demand and Dynamic Games", "Advances in Economics"),
    "ChowdhuryVerma2024": ("Chowdhury", "Walking a tightrope", "Production Economics"),
    "Fukasawa2025": ("Fukasawa", "When do firms sell high durability products", ""),
    "Brusselaers2020": ("Brusselaers", "Economic consequences of consumer repair strategies for electrical household devices", "Enterprise Information"),
    "Zurn2025": ("Zurn", "Financial incentives for the repair of consumer goods: their role in consumers' repair decision", ""),
}


def busca(autor, titulo, revista):
    q = urllib.parse.quote(f"{autor} {titulo}")
    url = f"https://api.crossref.org/works?rows=25&query.bibliographic={q}"
    for espera in (5, 20, 60, 120, 0):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=md.H), timeout=60) as r:
                items = json.load(r)["message"]["items"]
            break
        except Exception:
            if not espera:
                raise
            time.sleep(espera)
    time.sleep(3)
    mejores = []
    for it in items:
        t = (it.get("title") or [""])[0]
        c = (it.get("container-title") or [""])[0]
        s = v.sim(titulo, t)
        if s >= 0.6 and (not revista or revista.lower() in c.lower()):
            mejores.append((round(s, 2), it.get("DOI"), t[:70], c[:45], md.anio(it)))
    return sorted(mejores, reverse=True)[:3]


meta = json.loads((AQUI / "metadatos.json").read_text(encoding="utf-8"))
print("== búsqueda filtrada")
for k, (a, t, r) in PENDIENTES.items():
    res = busca(a, t, r)
    print(k, res)
    if res and res[0][0] >= 0.85:
        m = md.por_doi(res[0][1])
        if m:
            meta[k] = md.compacta(m)
            print("   -> resuelto", meta[k]["doi"], meta[k]["anio"], meta[k]["revista"], meta[k]["vol"], meta[k]["pag"])
    elif k in meta:
        print("   -> sin versión publicada, se conserva", meta[k]["doi"])
(AQUI / "metadatos.json").write_text(json.dumps(meta, ensure_ascii=False, indent=0), encoding="utf-8")

print("== aún preprint o reseña en metadatos")
for k, m in meta.items():
    if (m["doi"] or "").lower().startswith(("10.2139", "10.3386", "10.21034", "10.1920", "10.2172")):
        print(k, m["doi"], m["titulo"][:60])
    if m["tipo"] == "journal-article" and m["revista"] in ("Technometrics", "Southern Economic Journal", "Kyklos"):
        print("RESEÑA?", k, m["doi"], m["revista"])
    if not m["autores"]:
        print("SIN AUTORES", k, m["doi"])
    if m["tipo"] == "journal-article" and not (m["vol"] and m["pag"]):
        print("INCOMPLETO", k, m["doi"], m["revista"], m["anio"], m["vol"], m["pag"])

print("== tesis vs Crossref")
texto = BIB.read_text(encoding="utf-8")
for bloque in re.split(r"\n(?=@)", texto):
    c = re.match(r"@\w+\{(.+?),", bloque)
    if not c:
        continue
    k = c.group(1)
    if k not in meta:
        print("NO EN CATÁLOGO", k)
        continue
    campo = lambda n: (re.search(rf"^\s*{n}\s*=\s*\{{(.*)\}},", bloque, re.M) or [None, ""])[1]
    m = meta[k]
    pag_b = campo("pages").replace("--", "-")
    dif = []
    if campo("volume") and m["vol"] and campo("volume") != str(m["vol"]):
        dif.append(f"vol tesis={campo('volume')} crossref={m['vol']}")
    if pag_b and m["pag"]:
        pb = re.findall(r"\d+", pag_b)
        pm = re.findall(r"\d+", m["pag"])
        if pb[:1] != pm[:1]:
            dif.append(f"pág tesis={pag_b} crossref={m['pag']}")
    if campo("year") and m["anio"] and abs(int(campo("year")) - int(m["anio"])) > 0:
        dif.append(f"año tesis={campo('year')} crossref={m['anio']}")
    if campo("journal") and m["revista"] and v.sim(campo("journal"), m["revista"]) < 0.6:
        dif.append(f"revista tesis={campo('journal')} crossref={m['revista']}")
    if dif:
        print(k, "; ".join(dif))
