"""Baja de OpenAlex los resúmenes publicados de las obras con DOI y los deja en abstracts.txt (claves + 650 caracteres)."""
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
AQUI = pathlib.Path(__file__).parent
sys.path.insert(0, str(AQUI))
import catalogo as c  # noqa: E402

meta = json.loads((AQUI / "metadatos.json").read_text(encoding="utf-8"))
claves = [e["clave"] for _, _, subs, _ in c.SECCIONES for _, es in subs for e in es]
doi_de = {}
for k in claves:
    d = (meta.get(k) or {}).get("doi")
    if not d and k in c.MANUAL and "doi.org/" in c.MANUAL[k][1]:
        d = c.MANUAL[k][1].split("doi.org/")[1]
    if d:
        doi_de[k] = d.lower()
print(len(claves), "claves;", len(doi_de), "con DOI")


def texto(inv):
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


por_doi = {}
dois = sorted(set(doi_de.values()))
for i in range(0, len(dois), 40):
    lote = dois[i:i + 40]
    q = urllib.parse.urlencode({"filter": "doi:" + "|".join(lote), "per-page": 50, "select": "doi,abstract_inverted_index"})
    for intento in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request("https://api.openalex.org/works?" + q, headers={"User-Agent": "literatura-tesis/1.0"}), timeout=60) as r:
                for w in json.load(r)["results"]:
                    por_doi[(w["doi"] or "").replace("https://doi.org/", "").lower()] = texto(w.get("abstract_inverted_index"))
            break
        except Exception as e:
            print("reintento", i, e)
            time.sleep(5)
    time.sleep(0.5)

res = {k: por_doi.get(d, "") for k, d in doi_de.items()}
(AQUI / "abstracts.json").write_text(json.dumps(res, ensure_ascii=False, indent=0), encoding="utf-8")
con = {k: v for k, v in res.items() if v}
print(len(con), "con resumen")
with open(AQUI / "abstracts.txt", "w", encoding="utf-8") as f:
    for k in claves:
        if res.get(k):
            f.write(f"[{k}] {res[k][:650]}\n")
print("sin resumen:", [k for k in claves if not res.get(k)])
