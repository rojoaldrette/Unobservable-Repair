"""Verifica en Crossref que cada referencia de candidatas.txt exista.

Uso: python verificar.py
Escribe verificacion.csv con: clave, estado, similitud, año encontrado, DOI, autor y título encontrados.

estado:
  OK      título casi idéntico, apellido presente y año a +-2 (coincide la referencia)
  REVISAR parecido pero falla algo; hay que mirarla a mano
  NO      Crossref no tiene nada parecido (libros y documentos de trabajo suelen caer aquí)
"""
import csv
import difflib
import json
import pathlib
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

AQUI = pathlib.Path(__file__).parent
API = "https://api.crossref.org/works?rows=5&select=DOI,title,author,issued,container-title,type&query.bibliographic="
HEADERS = {"User-Agent": "literatura-tesis/1.0"}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", s).split()


def sim(a, b):
    return difflib.SequenceMatcher(None, " ".join(norm(a)), " ".join(norm(b))).ratio()


def consulta(q):
    url = API + urllib.parse.quote(q)
    for intento in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=40) as r:
                return json.load(r)["message"]["items"]
        except Exception:
            time.sleep(2 * (intento + 1))
    return None


def verifica(linea):
    clave, autor, anio, titulo = [x.strip() for x in linea.split("|", 3)]
    items = consulta(f"{autor} {titulo}")
    if items is None:
        return [clave, "ERROR", "", "", "", "", ""]
    mejor, puntos = None, -1.0
    for it in items:
        t = (it.get("title") or [""])[0]
        apellidos = " ".join(a.get("family", "") for a in it.get("author", []))
        a2 = ((it.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        s = sim(titulo, t)
        ok_autor = norm(autor)[0] in norm(apellidos) if norm(autor) else False
        ok_anio = a2 is not None and abs(a2 - int(anio)) <= 2
        p = s + 0.1 * ok_autor + 0.05 * ok_anio
        if p > puntos:
            puntos = p
            mejor = (t, s, ok_autor, ok_anio, a2, it.get("DOI", ""), apellidos, (it.get("container-title") or [""])[0])
    if mejor is None:
        return [clave, "NO", "", "", "", "", ""]
    t, s, oa, oy, a2, doi, ap, rev = mejor
    if s >= 0.85 and oa and oy:
        estado = "OK"
    elif s >= 0.6:
        estado = "REVISAR"
    else:
        estado = "NO"
    return [clave, estado, f"{s:.2f}", a2, doi, ap[:60], f"{t[:90]} [{rev[:40]}]"]


def main():
    lineas = [l for l in (AQUI / "candidatas.txt").read_text(encoding="utf-8").splitlines()
              if l.strip() and not l.startswith("#")]
    with ThreadPoolExecutor(6) as ex:
        filas = list(ex.map(verifica, lineas))
    with open(AQUI / "verificacion.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clave", "estado", "similitud", "anio", "doi", "autores", "titulo_encontrado"])
        w.writerows(filas)
    conteo = {}
    for f in filas:
        conteo[f[1]] = conteo.get(f[1], 0) + 1
    print(conteo)
    for f in filas:
        if f[1] != "OK":
            print(f[1], f[0], f[2], f[3], f[4], "||", f[5], "||", f[6])


if __name__ == "__main__":
    main()
