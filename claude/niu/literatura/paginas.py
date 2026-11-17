"""Completa rangos de páginas (Crossref da solo la primera en DOI de JSTOR) con OpenAlex y la bibliografía de la tesis."""
import json
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
AQUI = pathlib.Path(__file__).parent
meta = json.loads((AQUI / "metadatos.json").read_text(encoding="utf-8"))
bib = (AQUI.parent / "utils_tex" / "bib" / "referencias.bib").read_text(encoding="utf-8")
PAG_TESIS = {}
for bloque in re.split(r"\n(?=@)", bib):
    c = re.match(r"@\w+\{(.+?),", bloque)
    p = re.search(r"^\s*pages\s*=\s*\{(.*)\},", bloque, re.M)
    if c and p:
        PAG_TESIS[c.group(1)] = p.group(1).replace("--", "-")

solas = [k for k, m in meta.items() if re.fullmatch(r"\d+", str(m["pag"] or "")) and m["tipo"] == "journal-article"]
print(len(solas), "con una sola página")
sin_rango = []
for k in solas:
    m = meta[k]
    primera = str(m["pag"])
    if PAG_TESIS.get(k, "").startswith(primera + "-"):
        m["pag"] = PAG_TESIS[k]
        continue
    time.sleep(0.3)
    url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(m["doi"], safe="/") + "?select=biblio"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "literatura-tesis/1.0"}), timeout=40) as r:
            b = json.load(r).get("biblio") or {}
        fin = b.get("last_page")
        if b.get("first_page") == primera and fin and fin != primera and int(fin) > int(primera):
            m["pag"] = f"{primera}-{fin}"
            continue
    except Exception as e:
        print("  error", k, e)
    sin_rango.append((k, primera, m["doi"]))
print("sin rango (quedan con número de artículo o primera página):", sin_rango)
(AQUI / "metadatos.json").write_text(json.dumps(meta, ensure_ascii=False, indent=0), encoding="utf-8")
