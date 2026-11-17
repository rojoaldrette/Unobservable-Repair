"""Busca en Crossref las claves del catálogo que siguen sin metadatos y las agrega a metadatos.json."""
import json
import pathlib
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
AQUI = pathlib.Path(__file__).parent
sys.path.insert(0, str(AQUI))
import catalogo as c  # noqa: E402
import metadatos as md  # noqa: E402
import verificar as v  # noqa: E402

meta = json.loads((AQUI / "metadatos.json").read_text(encoding="utf-8"))
cand = {l.split("|")[0].strip(): [x.strip() for x in l.split("|", 3)]
        for l in (AQUI / "candidatas.txt").read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")}
claves = [e["clave"] for _, _, subs, _ in c.SECCIONES for _, es in subs for e in es]
faltan = [k for k in claves if k not in meta and k not in c.MANUAL]
print("sin datos:", faltan)
for k in faltan:
    if k not in cand:
        print("  sin candidata:", k)
        continue
    _, autor, _, titulo = cand[k]
    time.sleep(3)
    for it in v.consulta(f"{autor} {titulo}") or []:
        t = (it.get("title") or [""])[0]
        if v.sim(titulo, t) >= 0.85:
            m = md.por_doi(it["DOI"])
            if m:
                meta[k] = md.compacta(m)
                print("  ", k, meta[k]["doi"], meta[k]["anio"], meta[k]["revista"], meta[k]["autores"][:2])
                break
    else:
        print("  NO ENCONTRADA", k)
(AQUI / "metadatos.json").write_text(json.dumps(meta, ensure_ascii=False, indent=0), encoding="utf-8")
