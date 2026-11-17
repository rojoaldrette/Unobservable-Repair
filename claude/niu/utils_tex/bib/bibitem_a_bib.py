"""Convierte el bloque thebibliography de un .tex a un archivo .bib.

Uso:
    python bibitem_a_bib.py entrada.tex salida.bib

Reconoce los tres formatos que usa la tesis:
  artículo:   Autores (Año). ``Título''. \\textit{Revista}, vol(num), págs.
  libro:      Autores (Año). \\textit{Título}[, 2a. ed.]. Editorial.
  capítulo:   Autores (Año). ``Título''. En Editores (ed./eds.), \\textit{Libro}, [vol. N,] págs. Editorial.
Lo que no reconoce lo deja como @misc con note = resto del texto, para revisarlo a mano.
"""
import re
import sys


def split_names(s):
    """'Adda, J., R. Cooper y P. Mira' -> ['Adda, J.', 'R. Cooper', 'P. Mira']."""
    s = s.strip()
    others = False
    if s.endswith(" et al."):
        s, others = s[: -len(" et al.")], True
    m = re.search(r"\s(?:y|e)\s(?=\S+\.)", s)  # último conector antes de una inicial
    last = None
    if m:
        # usar la última ocurrencia
        ms = list(re.finditer(r"\s(?:y|e)\s(?=[A-Z][\w\-]*\.)", s))
        if ms:
            m = ms[-1]
            s, last = s[: m.start()], s[m.end():]
    parts = [p.strip() for p in s.split(", ")]
    names = []
    # Si el primer bloque viene "Apellido, Iniciales" (autores), juntar los dos primeros.
    if len(parts) >= 2 and not re.match(r"^[A-Z][\w\-]*\.", parts[0]) and re.match(r"^[A-Z][\w\-]*\.", parts[1]):
        names.append(parts[0] + ", " + parts[1])
        parts = parts[2:]
    names += parts
    if last:
        names.append(last.strip())
    if others:
        names.append("others")
    return [n for n in names if n]


def tex_accents(s):
    return (s.replace("ü", r"{\"u}").replace("ö", r"{\"o}").replace("é", r"{\'e}")
             .replace("á", r"{\'a}").replace("í", r"{\'i}").replace("ó", r"{\'o}"))


def parse_item(key, text):
    text = " ".join(text.split())
    m = re.match(r"(.+?) \((\d{4})([a-z]?)\)\. (.*)$", text)
    if not m:
        return "misc", {"note": text}
    authors, year, letra, rest = m.groups()
    f = {"author": " and ".join(split_names(authors)), "year": year}
    if letra:  # sin esto biblatex asigna a/b por orden de título y puede invertirlas
        f["sorttitle"] = letra

    # Libro: \textit{Título}[, N a. ed.]. Editorial.
    mb = re.match(r"\\textit\{(.+?)\}(?:, (\d+)a\. ed)?\.(.+?)\.?$", rest)
    if mb:
        f["title"] = mb.group(1)
        if mb.group(2):
            f["edition"] = mb.group(2)
        f["publisher"] = mb.group(3).strip()
        return "book", f

    mt = re.match(r"``(.+?)''\.? (.*)$", rest)
    if not mt:
        f["note"] = rest
        return "misc", f
    title, rest = mt.groups()
    f["title"] = title.replace("'\\,", "'").rstrip(".")

    # Capítulo de libro
    mc = re.match(r"En (.+?) \((?:ed|eds)\.\), \\textit\{(.+?)\}, (?:vol\. (\d+), )?(\d+)--(\d+)\. (.+?)\.?$", rest)
    if mc:
        eds, book, vol, p1, p2, pub = mc.groups()
        f["editor"] = " and ".join(split_names(eds))
        f["booktitle"] = book
        if vol:
            f["volume"] = vol
        f["pages"] = f"{p1}--{p2}"
        f["publisher"] = pub
        return "incollection", f

    # Artículo
    ma = re.match(r"\\textit\{(.+?)\}, (\d+)(?:\((\d+)\))?, ([\d\-]+)\.?$", rest)
    if ma:
        journal, vol, num, pages = ma.groups()
        f["journal"] = journal
        f["volume"] = vol
        if num:
            f["number"] = num
        f["pages"] = pages
        return "article", f

    # arXiv / documento de trabajo
    mx = re.match(r"Documento de trabajo, arXiv:(\d+\.\d+)\.?$", rest)
    if mx:
        f["note"] = "Documento de trabajo"
        f["eprint"] = mx.group(1)
        f["eprinttype"] = "arxiv"
        return "misc", f

    f["note"] = rest
    return "misc", f


ORDER = ["author", "editor", "year", "title", "journal", "booktitle", "volume", "number",
         "pages", "edition", "publisher", "eprint", "eprinttype", "note", "sorttitle"]


def main(src, dst):
    tex = open(src, encoding="utf-8").read()
    block = re.search(r"\\begin\{thebibliography\}\{.*?\}(.*?)\\end\{thebibliography\}", tex, re.S).group(1)
    items = re.findall(r"\\bibitem\{(.+?)\}(.*?)(?=\\bibitem\{|\Z)", block, re.S)
    out, revisar = [], []
    for key, body in items:
        kind, f = parse_item(key, body)
        if kind == "misc" and "eprint" not in f:
            revisar.append(key)
        lines = [f"@{kind}{{{key},"]
        for k in ORDER:
            if k in f:
                lines.append(f"  {k:<10} = {{{tex_accents(f[k])}}},")
        lines.append("}")
        out.append("\n".join(lines))
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(out) + "\n")
    print(f"{len(items)} entradas -> {dst}")
    if revisar:
        print("Revisar a mano:", ", ".join(revisar))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
