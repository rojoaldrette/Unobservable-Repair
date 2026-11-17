"""Baja por DOI los metadatos de cada referencia y comprueba los enlaces sin DOI.

Salida: metadatos.json (clave -> campos de Crossref) y enlaces.csv (url, código HTTP, título de la página).
Avisa si el título del DOI no coincide con el título candidato: así un DOI mal recordado no pasa.
"""
import csv
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import verificar as v  # noqa: E402

AQUI = pathlib.Path(__file__).parent
H = {"User-Agent": "literatura-tesis/1.0"}

# Versión publicada en vez de preprint o reseña; se comprueba el título de cada una.
DOI = {
    "Pakes1986": "10.2307/1912835", "AguirregabiriaNevo2013": "10.1017/CBO9781139060035.004",
    "KeaneWolpin1994": "10.2307/2109768", "KeaneWolpin1997": "10.1086/262080",
    "DubeFoxSu2012": "10.3982/ECTA8585", "NoretsTang2014": "10.1093/restud/rdt050",
    "ChiongGalichonShum2016": "10.3982/QE436", "HuShumTanXiao2017": "10.1515/jem-2015-0016",
    "Rabiner1989": "10.1109/5.18626", "CunhaHeckmanSchennach2010": "10.3982/ECTA6551",
    "MisraNair2011": "10.1007/s11129-011-9096-1", "GavazzaLizzeriRoketskiy2014": "10.1257/aer.104.11.3668",
    "Schiraldi2011": "10.1111/j.1756-2171.2011.00134.x", "Blundell2026": "10.3982/QE1822",
    "Carranza2010": "10.1016/j.ijindorg.2010.02.003", "Nair2007": "10.1007/s11129-007-9026-4",
    "HendelNevo2006": "10.1111/j.1468-0262.2006.00719.x", "GoettlerGordon2011": "10.1086/664615",
    "Ghose2009": "10.2307/20650292", "AllcottWozny2014": "10.1162/REST_a_00419",
    "LiTimminsvH2009": "10.1257/pol.1.2.113", "JacobsenVanBenthem2015": "10.1257/aer.20130935",
    "MianSufi2012": "10.1093/qje/qjs024", "Petrin2002": "10.1086/340779", "Aiyagari1994": "10.2307/2118417",
    "PakesMcGuire1994": "10.2307/2555975", "DubeHC2010": "10.1287/mksc.1090.0541",
    "HendelLizzeri1999b": "10.1257/aer.89.5.1097", "LaceteraPopeSydnor2012": "10.1257/aer.102.5.2206",
    "Puterman1994": "10.1002/9780470316887", "Byrd1995": "10.1137/0916069",
    "Aldrich2011": "10.1016/j.jedc.2010.10.001", "Kirchherr2017": "10.1016/j.resconrec.2017.09.005",
    "SuJudd2012": "10.3982/ECTA7925", "KasaharaShimotsu2012": "10.3982/ECTA8291",
    "Train2009": "10.1017/CBO9780511805271", "ChevalierGoolsbee2009": "10.1162/qjec.2009.124.4.1853",
    "WeintraubBVR2008": "10.3982/ECTA6158", "HouseLeahy2004": "10.1086/383101",
    "ChenRoss1993": "10.1111/j.1430-9134.1993.00593.x", "HMMMaintenance2022": "10.1016/j.ejor.2022.05.006",
    "NocedalWright2006": "10.1007/978-0-387-40065-5", "Cooper2010": "10.4324/9781315592930",
    "Borenstein2000": "10.1111/j.1430-9134.2000.00141.x", "Melnikov2013": "10.1111/j.1465-7295.2012.00501.x",
    "Bellman1957": "10.1515/9781400835386", "Rust1988": "10.1137/0326056",
    "ArcidiaconoMiller2011": "10.3982/ecta7743", "Norets2009": "10.3982/ecta7292",
    "ImaiJainChing2009": "10.3982/ecta5658", "KasaharaShimotsu2009": "10.3982/ecta6763",
    "PerrigneVuong2011": "10.3982/ecta6954", "Bond1982x": "", "RightToRepairRRC2025": "10.1007/s11149-025-09490-z",
    "ChenGMW2019": "10.1111/roie.12436", "Perzanowski2022": "10.1017/9781108946926",
    "GriewankWalther2008": "10.1137/1.9780898717761", "Slade2006": "10.4159/9780674043756",
    "Huisman2012jie": "10.1111/jiec.12012", "Schreiber2022": "10.24857/rgsa.v16n2-023",
    "OrtegaRheinboldt1970": "10.1137/1.9780898719468", "OhtaGriliches1976": "10.4337/9781035351442.00029",
}
# Referencias nuevas por DOI directo o búsqueda (salieron de la bibliografía de Waldman 2003 y de la búsqueda web).
NUEVAS = [
    "Schmalensee1974|Schmalensee|1974|Market Structure, Durability, and Maintenance Effort",
    "Shapiro1995|Shapiro|1995|Aftermarkets and Consumer Welfare: Making Sense of Kodak",
    "Sobel1991|Sobel|1991|Durable Goods Monopoly with Entry of New Consumers",
    "LevinthalPurohit1989|Levinthal|1989|Durable Goods and Product Obsolescence",
    "Waldman1996b|Waldman|1996|Durable Goods Pricing When Quality Matters",
    "Su1975|Su|1975|Durability of Consumption Goods Reconsidered",
    "DeOliveira2013b|Oliveira|2013|Planned Obsolescence Leadership and Management in Engineering",
    "ChowdhuryVerma2024b|Chowdhury|2024|Walking a tightrope: Impact of adopting voluntary right-to-repair policy on firm profit and consumer welfare",
]
URLS = [
    "https://eml.berkeley.edu/reprints/mcfadden/zarembka.pdf",
    "https://cowles.yale.edu/sites/default/files/2022-12/d0477.pdf",
    "https://eml.berkeley.edu/~mcfadden/discrete.html",
    "https://mitpress.mit.edu/9780262100717/numerical-methods-in-economics/",
    "https://mitpress.mit.edu/9780262232586/econometric-analysis-of-cross-section-and-panel-data/",
    "https://mitpress.mit.edu/9780262134200/applied-computational-economics-and-finance/",
    "https://mitpress.mit.edu/9780262012010/dynamic-economics/",
    "https://www.jmlr.org/papers/v18/17-468.html",
    "https://www.nber.org/chapters/c10206",
    "https://www.nber.org/chapters/c6492",
    "https://www.nber.org/papers/w34264",
    "https://www.nber.org/papers/w7785",
    "https://www.nber.org/papers/w25840",
    "https://www.nber.org/books-and-chapters/measurement-durable-goods-prices",
    "https://arxiv.org/abs/2503.23792",
    "https://arxiv.org/abs/2105.15183",
    "https://eur-lex.europa.eu/eli/dir/2024/1799/oj",
    "https://eur-lex.europa.eu/eli/reg/2024/1781/oj",
    "https://econpapers.repec.org/RePEc:bdm:wpaper:2022-02",
    "https://ideas.repec.org/a/aea/aecrev/v60y1970i5p884-94.html",
    "https://ideas.repec.org/a/aea/aecrev/v59y1969i1p102-07.html",
    "https://ideas.repec.org/a/aea/aecrev/v72y1982i4p816-24.html",
    "https://ideas.repec.org/a/aea/aecrev/v72y1982i4p836-40.html",
    "https://web.mit.edu/15.053/www/AMP-Chapter-11.pdf",
    "https://www.jstor.org/stable/23830511",
    "https://openyls.law.yale.edu/handle/20.500.13051/8128",
    "https://hal.science/hal-04535706",
    "https://ewastemonitor.info/gem-2020/",
    "https://doi.org/10.5089/9781513545851.001",
    "https://github.com/jax-ml/jax",
    "https://www.rff.org/publications/working-papers/competition-and-car-longevity/",
    "https://www.minneapolisfed.org/research/quarterly-review/maintenance-and-repair-too-big-to-ignore",
    "https://www.nber.org/papers/w24561",
    "https://www.aeaweb.org/conference/2018/preliminary/paper/B6dbT2kD",
    "https://www.nber.org/papers/w26281",
    "https://lirias.kuleuven.be/retrieve/0d654c24-7243-430b-af2d-d7a3713ec376",
]


def por_doi(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=40) as r:
                return json.load(r)["message"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return None


def anio(m):
    for c in ("published-print", "journal-issue", "published", "issued"):
        d = (m.get(c) or {})
        if c == "journal-issue":
            d = (d.get("published-print") or {})
        p = (d.get("date-parts") or [[None]])[0][0]
        if p:
            return p
    return None


def compacta(m):
    return {
        "doi": m.get("DOI"),
        "titulo": (m.get("title") or [""])[0],
        "autores": [(a.get("family") or a.get("name") or "", a.get("given", "")) for a in m.get("author", [])],
        "anio": anio(m),
        "revista": (m.get("container-title") or [""])[0],
        "vol": m.get("volume", ""), "num": m.get("issue", ""), "pag": m.get("page", ""),
        "tipo": m.get("type", ""), "editorial": m.get("publisher", ""),
    }


def http(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (literatura-tesis)"})
        with urllib.request.urlopen(req, timeout=30) as r:
            t = r.read(200000)
            m = re.search(rb"<title[^>]*>(.*?)</title>", t, re.S | re.I)
            ti = m.group(1).decode("utf-8", "replace").strip()[:100] if m else ("PDF" if t[:4] == b"%PDF" else "")
            return [url, r.status, ti]
    except urllib.error.HTTPError as e:
        return [url, e.code, ""]
    except Exception as e:
        return [url, "ERR", str(e)[:60]]


def main():
    filas = {f[0]: f for f in list(csv.reader(open(AQUI / "verificacion.csv", encoding="utf-8")))[1:]}
    cand = {l.split("|")[0].strip(): [x.strip() for x in l.split("|", 3)]
            for l in (AQUI / "candidatas.txt").read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")}
    for l in NUEVAS:
        c = [x.strip() for x in l.split("|", 3)]
        cand[c[0]] = c
        filas[c[0]] = v.verifica(l)
        print("NUEVA", filas[c[0]])

    def tarea(k):
        doi = DOI.get(k)
        f = filas[k]
        if not doi and f[1] == "OK":
            doi = f[4]
        if not doi and k in [x.split("|")[0] for x in NUEVAS] and f[4]:
            doi = f[4]
        if not doi:
            return k, None, "sin DOI"
        m = por_doi(doi)
        if m is None:
            return k, None, f"DOI inválido {doi}"
        return k, compacta(m), ""

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(tarea, cand))
    out = {}
    print("== problemas")
    for k, m, nota in res:
        if m is None:
            print(k, nota, "|", filas[k][1], filas[k][4])
            continue
        out[k] = m
        s = v.sim(cand[k][3], m["titulo"])
        ok_a = any(v.norm(cand[k][1])[0] in v.norm(a[0]) for a in m["autores"]) if m["autores"] else None
        if s < 0.8 or ok_a is False or not m["revista"] and m["tipo"] == "journal-article":
            print(k, f"sim={s:.2f}", "autor", ok_a, m["anio"], m["doi"], "||", m["titulo"][:80], "||", m["revista"][:40])
    (AQUI / "metadatos.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(len(out), "con metadatos")
    with ThreadPoolExecutor(6) as ex:
        en = list(ex.map(http, URLS))
    with open(AQUI / "enlaces.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(en)
    print("== enlaces")
    for e in en:
        print(e[1], e[0], "||", e[2])


if __name__ == "__main__":
    main()
