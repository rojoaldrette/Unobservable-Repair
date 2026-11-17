"""Segundo pase: reintenta errores de red y lista casos dudosos entre los OK. Actualiza verificacion.csv."""
import csv
import pathlib
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import verificar as v  # noqa: E402

AQUI = pathlib.Path(__file__).parent
filas = list(csv.reader(open(AQUI / "verificacion.csv", encoding="utf-8")))
cab, filas = filas[0], filas[1:]
lineas = {l.split("|")[0].strip(): l for l in (AQUI / "candidatas.txt").read_text(encoding="utf-8").splitlines()
          if l.strip() and not l.startswith("#")}
for i, f in enumerate(filas):
    if f[1] == "ERROR":
        time.sleep(1)
        filas[i] = v.verifica(lineas[f[0]])
with open(AQUI / "verificacion.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(cab)
    w.writerows(filas)

print("== antes REVISAR/NO/ERROR, faltaban en la salida")
vistos = {"McFadden", }
for f in filas:
    if f[1] in ("ERROR",):
        print(f)
print("== reintentados y resto de no OK desde Cooper2010")
imprime = False
for f in filas:
    if f[0] == "ChenGMW2019":
        imprime = True
    if imprime and f[1] != "OK":
        print(f[1], f[0], f[2], f[3], f[4], "||", f[5], "||", f[6])
print("== reintentados")
for clave in ["KeaneToddWolpin2011", "KSS2021b", "AllmanMatiasRhodes2009", "Berkovec1985", "Attanasio2000",
              "GreenspanCohen1999", "Young2010", "DulleckKerschbamer2006", "Borenstein2000", "Fachbach2022",
              "DrinkwaterHastings1967", "Howard1960", "MirandaFackler2002", "WalkerNi2011", "NelderMead1965",
              "OguchiFuse2015"]:
    f = next(x for x in filas if x[0] == clave)
    print(f[1], f[0], f[2], f[3], f[4], "||", f[5], "||", f[6])

print("== OK sospechosos: preprint, sin revista o año distinto")
anio = {k: int(l.split("|")[2]) for k, l in lineas.items()}
for f in filas:
    if f[1] != "OK":
        continue
    doi = f[4]
    sin_revista = f[6].rstrip().endswith("[]")
    if doi.startswith(("10.2139", "10.3386", "10.1920")) or sin_revista or (f[3] and int(f[3]) != anio[f[0]]):
        print(f[0], "cand", anio[f[0]], "hallado", f[3], doi, "||", f[6])
