"""Genera literatura_tesis.tex a partir de catalogo.py y metadatos.json.

Uso: python generar.py
Avisa de claves sin datos, repetidas o de metadatos sin usar.
"""
import html
import json
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
AQUI = pathlib.Path(__file__).parent
sys.path.insert(0, str(AQUI))
import catalogo as c  # noqa: E402

META = json.loads((AQUI / "metadatos.json").read_text(encoding="utf-8"))
TESIS = (AQUI.parent / "tesis_claude" / "tesis.tex").read_text(encoding="utf-8")
EN_TESIS = set(re.findall(r"\\bibitem\{(.+?)\}", TESIS))
# Rangos tomados de la bibliografía de Waldman (2003) y de la portada de tus PDF.
PAGINAS = {
    "Stokey1981": "112-128", "Swan1971": "347-357", "KleimanOphir1966": "165-178", "Schmalensee1970": "54-64",
    "Schmalensee1974": "277-287", "SieperSwan1973": "333-351", "BondSamuelson1984": "336-345", "Sobel1991": "1455-1485",
    "Waldman1996b": "489-510", "GuptaRomano1998": "427-442", "Parks1977": "1099-1115",
}
MENORES = {"of", "and", "the", "in", "on", "for", "a", "an", "to", "with", "from", "by", "at", "as"}


def esc(s):
    s = html.unescape(s or "")
    s = re.sub(r"</?[a-zA-Z:]+[^>]*>", "", s)
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    for a, b in (("\\", ""), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("$", "\\$")):
        s = s.replace(a, b)
    return s.replace("{", "").replace("}", "")


def texto_libre(s):
    """Resúmenes: conserva \\% escrito a mano y escapa lo demás."""
    s = s.replace("\\%", "\0")
    s = esc(s)
    return s.replace("\0", "\\%").replace("\\%", "\\%")


def titulo(s):
    s = esc(s)
    letras = [x for x in s if x.isalpha()]
    if letras and sum(x.isupper() for x in letras) / len(letras) > 0.8:
        ps = s.lower().split()
        s = " ".join(p if i and p in MENORES else p[:1].upper() + p[1:] for i, p in enumerate(ps))
    s = re.sub(r"^Chapter \d+\.? ", "", s)
    for a, b in (("U.s.", "U.S."), ("and firm Dynamics", "and Firm Dynamics"), ("Us Nuclear", "US Nuclear")):
        s = s.replace(a, b)
    return s


def inicial(g):
    trozos = []
    for p in g.split():
        partes = p.split("-")
        trozos.append("-".join(x[0].upper() + "." for x in partes if x))
    return " ".join(trozos)


def apellido(f):
    return f.title() if f.isupper() else f


def autores(k, m):
    if k in c.AUTORES:
        return c.AUTORES[k] + ("." if c.AUTORES[k].endswith(("Rust, J",)) else "")
    a = m["autores"]
    if not a:
        return ""
    if len(a) > 8:
        return f"{apellido(a[0][0])}, {inicial(a[0][1])} et al."
    fmt = [f"{apellido(a[0][0])}, {inicial(a[0][1])}".rstrip(", ")]
    fmt += [f"{inicial(g)} {apellido(f)}".strip() for f, g in a[1:]]
    if len(fmt) == 1:
        return fmt[0]
    conector = " e " if fmt[-1][:1] in "IiYy" and fmt[-1][:2] not in ("Hi", "hi") else " y "
    return ", ".join(fmt[:-1]) + conector + fmt[-1]


def referencia(k):
    """Devuelve (texto LaTeX de la cita, url, cómo se verificó)."""
    if k in c.MANUAL:
        t, u, v = c.MANUAL[k]
        return t, u, v
    m = META[k]
    an = c.ANIO.get(k, m["anio"])
    t = f"{esc(autores(k, m))} ({an}). "
    tipo = m["tipo"]
    if tipo in ("book", "monograph", "edited-book", "reference-book", "book-set"):
        t += f"\\textit{{{titulo(m['titulo'])}}}. {esc(m['editorial'])}."
    else:
        t += f"``{titulo(m['titulo'])}''. "
        if m["revista"]:
            prep = "En " if tipo in ("book-chapter", "book-section", "reference-entry") else ""
            t += f"{prep}\\textit{{{esc(m['revista'])}}}"
            v = ""
            if m["vol"]:
                v = f" {esc(str(m['vol']))}" + (f"({esc(str(m['num']))})" if m["num"] else "")
            pag = PAGINAS.get(k, str(m["pag"] or ""))
            if pag:
                solo = re.fullmatch(r"\d+", pag) and int(pag) < 100000 and tipo == "journal-article"
                v += ", " + ("desde p. " if solo else "") + esc(pag).replace("-", "--")
            t += v + "."
        elif tipo == "posted-content" or (m["doi"] or "").startswith(("10.2139", "10.3386")):
            serie = {"10.2139": "SSRN", "10.3386": "NBER"}.get((m["doi"] or "")[:7], "")
            t += f"Documento de trabajo {serie}.".replace(" .", ".")
        else:
            t += f"{esc(m['editorial'])}."
    return t, f"https://doi.org/{m['doi']}", "Crossref"


def enlace(u, v):
    if not u:
        return f"\\textit{{({esc(v)})}}"
    eti = u.replace("https://doi.org/", "doi:") if "doi.org" in u else u.replace("https://", "").replace("http://", "")
    eti = eti[:62] + ("…" if len(eti) > 62 else "")
    u_tex = u.replace("%", "\\%").replace("#", "\\#").replace("_", "\\_")
    nota = "" if v == "Crossref" else f" \\textit{{({esc(v)})}}"
    return f"\\href{{{u_tex}}}{{\\texttt{{{esc(eti)}}}}}{nota}"


def item(e, tengo, usados, n):
    k = e["clave"]
    if k in usados:
        print("REPETIDA", k)
    usados.add(k)
    t, u, v = referencia(k)
    marcas = ""
    if not tengo and e.get("prioridad"):
        marcas += "$\\star$\\,"
        n["prior"] += 1
    if k in EN_TESIS:
        marcas += "$\\dagger$\\,"
    if tengo and e.get("escaneado"):
        marcas += "$\\ddagger$\\,"
    s = f"\\item {marcas}{t} {enlace(u, v)}\\\\\n{texto_libre(e['resumen'])}"
    if tengo and e.get("archivo"):
        s += f"\\\\\n{{\\footnotesize Archivo: \\texttt{{{esc(e['archivo'])}}}}}"
    n["tengo" if tengo else "falta"] += 1
    return s


PRE = r"""% Generado por claude/niu/literatura/generar.py a partir de catalogo.py. Edita catalogo.py o este archivo.
\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[spanish,es-noshorthands]{babel}
\usepackage[margin=2.4cm]{geometry}
\usepackage{lmodern,microtype,enumitem,amssymb}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\setlist[itemize]{leftmargin=1.2em,itemsep=0.55em,topsep=0.4em,label={}}
\setlength{\parindent}{0pt}\setlength{\parskip}{0.5em}
\sloppy
\title{Mapa de literatura para la tesis:\\ equilibrio del mercado de autos con reparación no observada}
\author{Preparado por Claude para Rodrigo Aldrette}
\date{8 de noviembre de 2026}
\begin{document}
\maketitle
"""


def main():
    usados, n = set(), {"tengo": 0, "falta": 0, "prior": 0}
    cuerpo = []

    def nido(tit, intro, subs, transicion):
        cuerpo.append(f"\\section{{{esc(tit)}}}\n{texto_libre(intro)}\n")
        for sub, entradas in subs:
            cuerpo.append(f"\\subsection*{{{esc(sub)}}}\n\\addcontentsline{{toc}}{{subsection}}{{{esc(sub)}}}\n\\begin{{itemize}}")
            for e in entradas:
                cuerpo.append(item(e, "archivo" in e, usados, n))
            cuerpo.append("\\end{itemize}\n")
        if transicion:
            cuerpo.append(f"\\medskip\n\\noindent\\textit{{{texto_libre(transicion)}}}\n")

    cuerpo.append("\\part{Las seis subsecciones de la revisión de literatura}")
    for s in c.REVISION:
        nido(*s)
    cuerpo.append("\\part{Material de apoyo, fuera de la revisión}")
    for s in c.APOYO:
        nido(*s)
    intro = rf"""
\section*{{Cómo leer este documento}}
Clasifiqué {n['tengo']} obras de tu carpeta \texttt{{po}} y agregué {n['falta']} que faltan. La parte I sigue las seis subsecciones de la revisión, orientadas a tu pregunta: qué tanto importa la reparación en estos modelos. La parte II guarda lo que se cita en otro lado (métodos, motivación) o no se cita. Cada sección termina con lo que falta y, en la parte I, con una transición hacia la siguiente.

\begin{{itemize}}
\item $\star$ Prioridad: las {n['prior']} faltantes que más te conviene leer primero.
\item $\dagger$ Ya aparece en la bibliografía actualizada de la versión de Claude de la tesis (\texttt{{claude/niu/tesis\_claude/tesis.tex}}).
\item $\ddagger$ PDF escaneado sin texto extraíble: verifiqué la portada, pero el resumen viene de lo que sé de la obra.
\end{{itemize}}

\textbf{{Qué tan fiable es cada parte.}}
\begin{{itemize}}
\item \textbf{{Existencia:}} los datos con \texttt{{doi:}} se descargaron de Crossref y su título se comparó con la obra buscada. Los demás enlaces llevan entre paréntesis cómo se comprobaron. Quedan sin enlace dos reportes neerlandeses y dos documentos sin publicación, tomados de tus propios PDF.
\item \textbf{{Resúmenes de lo que tienes:}} salen de leer resumen o primeras páginas de cada PDF, no el artículo completo.
\item \textbf{{Resúmenes de lo que falta:}} cuando existe, cotejé el resumen publicado en OpenAlex (unas 180 de las 320 obras, contando ambos grupos). Del resto, salen de mi conocimiento y de búsquedas web. En ningún caso leí el texto completo. Revisa cualquier afirmación específica antes de citarla.
\item \textbf{{Contexto:}} es mi lectura de cómo se relaciona la obra con tu tesis, no algo que diga el autor.
\item \textbf{{Páginas:}} «desde p.» indica que solo pude confirmar la página inicial; completa el rango antes de citar.
\item \textbf{{Enlaces:}} el DOI asegura que la obra existe, pero puede pedirte acceso institucional.
\end{{itemize}}
\tableofcontents
\clearpage
"""
    apendice = r"""
\appendix
\section{Qué dejé fuera de la clasificación}
\begin{itemize}
\item \textbf{Trabajo tuyo, no literatura:} \texttt{Materia Julen/}, \texttt{Proyecto Aurora stuff/}, \texttt{output/} y \texttt{Fuentes/Thesis\_presentation.pdf}.
\item \textbf{Datos y fichas:} \texttt{Manual\_usuario\_EVSM.pdf} (escaneado, no lo pude leer) y el folleto comercial del \textit{IDC Mobile Phone Tracker}.
\item \textbf{Capítulos de Gordon (1990) resumidos en conjunto:} prefacio, aeronaves, equipo eléctrico, computadoras y electrodomésticos.
\item \textbf{Copias duplicadas:} las versiones repetidas de Moral y Jaumandreu; Arcidiacono y Miller; Gavazza et al.; Esteban y Shum; Gillingham et al.; y Bakker et al. Aparecen una sola vez.
\end{itemize}
\end{document}
"""
    (AQUI / "literatura_tesis.tex").write_text(PRE + intro + "\n".join(cuerpo) + apendice, encoding="utf-8")
    print(n)
    print("sin usar:", sorted(set(META) - usados - {"Zurn2025x"}))


if __name__ == "__main__":
    main()
