"""Convert draft/paper.md -> latex/main.tex (ACL style). Tailored to this paper's
markdown patterns. Iterative: compile and fix remaining issues by hand.
"""
from __future__ import annotations
import re
from pathlib import Path

SRC = Path("draft/paper.md")
OUT = Path("latex/main.tex")

UNI = {  # unicode -> LaTeX (math where needed)
    "κ": r"$\kappa$", "Δ": r"$\Delta$", "σ": r"$\sigma$", "μ": r"$\mu$",
    "≈": r"$\approx$", "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$",
    "→": r"$\rightarrow$", "↛": r"$\not\rightarrow$", "≠": r"$\neq$",
    "—": "---", "–": "--", "‘": "`", "’": "'", "“": "``", "”": "''",
    "…": r"\ldots{}", "✓": r"\checkmark{}", "✗": r"$\times$", "≅": r"$\approx$",
    "ﬁ": "fi", "ï": r'\"i', "é": r"\'e", "ö": r'\"o', "ü": r'\"u', "ä": r'\"a',
    "𝑎": "a", "√": r"$\surd$", "⁻": r"$^{-}$",
}

SPECIAL = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}

# verbatim (lstlisting) is not escaped, but listings chokes on multibyte UTF-8;
# map the few non-ASCII chars that appear inside the prompt blocks to ASCII.
CODE_ASCII = {"—": "--", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", "×": "x"}


def ascii_code(s: str) -> str:
    for u, t in CODE_ASCII.items():
        s = s.replace(u, t)
    return s


def esc_text(s: str) -> str:
    """Escape a run of plain text (no markdown/cite/code)."""
    for u, t in UNI.items():
        s = s.replace(u, t)
    out = []
    for ch in s:
        out.append(SPECIAL.get(ch, ch))
    s = "".join(out)
    # comparison operators in text -> math
    s = re.sub(r"(?<=\s)<(?=\s)", r"$<$", s)
    s = re.sub(r"(?<=\s)>(?=\s)", r"$>$", s)
    return s


def inline(s: str) -> str:
    """Process one line of inline markdown: protect cites + code, escape, then bold."""
    cites, codes = [], []
    def stash_cite(m):
        cites.append(m.group(0)); return f"\x00C{len(cites)-1}\x00"
    s = re.sub(r"\\cite[pt]\{[^}]*\}", stash_cite, s)
    def stash_code(m):
        codes.append(m.group(1)); return f"\x00K{len(codes)-1}\x00"
    s = re.sub(r"`([^`]*)`", stash_code, s)
    s = esc_text(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*([^*]+?)\*(?![\w*])", r"\\emph{\1}", s)
    # cross-references -> \ref so LaTeX auto-numbers (lets us cut/move tables safely)
    s = re.sub(r"\bTable (\d+)", r"Table~\\ref{tab:\1}", s)
    s = re.sub(r"\bFigure 1\b", r"Figure~\\ref{fig:worked}", s)
    s = re.sub(r"\bFigure 2\b", r"Figure~\\ref{fig:2x2}", s)
    for i, c in enumerate(codes):
        s = s.replace(f"\x00K{i}\x00", r"\texttt{\detokenize{" + c + "}}")
    for i, c in enumerate(cites):
        s = s.replace(f"\x00C{i}\x00", c)
    return s


def structural(s):
    """True if a stripped line begins a non-prose block, so paragraph accumulation
    (which lets **bold**/*emph* span soft-wrapped line breaks) must stop here."""
    return (not s
            or s[0] in "#|>"
            or s.startswith("- ")
            or s.startswith("```")
            or s.startswith("*[")
            or s == "---"
            or s.startswith("**Keywords:**")
            or s.startswith("**Anonymous submission")
            or bool(re.match(r"\*\*Table \d", s)))


def table(rows, caption, appendix=False):
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    body = [[c.strip() for c in r.strip("|").split("|")] for r in rows[2:]]
    ncol = len(header)
    maxlen = [0] * ncol
    for r in [header] + body:
        for j in range(min(ncol, len(r))):
            maxlen[j] = max(maxlen[j], len(r[j]))
    # A \FloatBarrier before \appendix keeps appendix floats out of the references.
    prose = max(maxlen) > 40                       # a long-text table (e.g. shipped-tools)
    cap_l = caption.lower()
    if "cohen" in cap_l:                            # kappa matrix: compact single column, body §5.2
        env, place, colsep, size = "table", "[t]", "2pt", "\\scriptsize"
        align = "l" + "c" * (ncol - 1)
    elif "shipped faithfulness judges" in cap_l:   # deployed-tools: anchored single column in appendix
        env, place, colsep, size = "table", "[H]", "3pt", "\\footnotesize"
        align = "p{1.4cm}p{5.6cm}"                 # Tool | Shipped method (2 cols)
    elif appendix and not prose:                    # numeric appendix table: single column
        env, place, colsep, size = "table", "[t]", "3pt", "\\scriptsize"
        align = "l" + "c" * (ncol - 1)
    elif prose:                                     # full-width wrapping text table
        env, place, colsep, size = "table*", "[t]", "4pt", "\\footnotesize"
        total = sum(maxlen) or 1
        align = "".join(f"p{{{max(1.2, 13.8*maxlen[j]/total):.1f}cm}}" for j in range(ncol))
    else:
        star = ncol >= 5 or sum(maxlen) > 50
        env, place, colsep, size = ("table*" if star else "table"), "[t]", "4pt", "\\footnotesize"
        align = "l" + "c" * (ncol - 1)
    out = [f"\\begin{{{env}}}{place}\n\\centering{size}\\setlength{{\\tabcolsep}}{{{colsep}}}",
           f"\\begin{{tabular}}{{{align}}}", "\\toprule",
           " & ".join(inline(h) for h in header) + r" \\", "\\midrule"]
    for r in body:
        r = (r + [""] * ncol)[:ncol]
        out.append(" & ".join(inline(c) for c in r) + r" \\")
    out += ["\\bottomrule", "\\end{tabular}",
            f"\\caption{{{caption}}}", f"\\end{{{env}}}"]
    return "\n".join(out)


def main():
    text = SRC.read_text()
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # strip comment header
    lines = text.split("\n")

    body = []
    i = 0
    title = "Citation Attribution is a Blind Spot for RAG Faithfulness Judges: An Audit of Seven Evaluators"
    pending_caption = None
    in_appendix = False
    in_references = False

    while i < len(lines):
        ln = lines[i]
        st = ln.strip()

        # stop at DRAFT-FLAGS trailing block
        if st.startswith("## DRAFT-FLAGS"):
            break
        # skip any *[ ... ]* editorial note, possibly spanning multiple lines
        if st.startswith("*["):
            while i < len(lines) and not lines[i].rstrip().endswith("]*"):
                i += 1
            i += 1; continue
        # skip anonymous-submission line, hrules, keywords
        if st.startswith("**Anonymous submission") or st == "---" or st.startswith("**Keywords:**"):
            i += 1; continue

        # title line
        if st.startswith("# "):
            i += 1; continue

        # code block -> verbatim
        if st.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i]); i += 1
            i += 1
            body.append("\\begin{lstlisting}\n" + ascii_code("\n".join(block)) + "\n\\end{lstlisting}")
            continue

        # tables: a line of "|...|" followed by "|---|"
        if st.startswith("|") and i + 1 < len(lines) and re.match(r"\|[-: |]+\|", lines[i+1].strip()):
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i]); i += 1
            cap = pending_caption or "Results."
            ncol = len([c for c in tbl[0].strip("|").split("|")])
            body.append(table(tbl, cap, appendix=in_appendix))
            pending_caption = None
            continue

        # capture a "**Table N. ...**" caption line (may span next line until blank)
        m = re.match(r"\*\*Table \d+\.(.*)", st)
        if m:
            caplines = [st]
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith("|"):
                caplines.append(lines[j].strip()); j += 1
            cap_md = " ".join(caplines)
            cap_md = re.sub(r"\s*Source:.*$", "", cap_md)   # drop internal provenance
            label = re.match(r"\*\*Table (\d+)\.", cap_md).group(1)
            cap_md = re.sub(r"^\*\*Table \d+\.\s*", "", cap_md).rstrip("*").strip().rstrip("*")
            cap_md = cap_md.replace("**", "")
            pending_caption = inline(cap_md) + f"\\label{{tab:{label}}}"
            i = j
            continue

        # figure blockquote stubs
        if st.startswith("> **[Figure 1"):
            while i < len(lines) and lines[i].strip().startswith(">"):
                i += 1
            body.append("%FIG1%")
            continue
        if st.startswith("> **[Figure 2"):
            while i < len(lines) and lines[i].strip().startswith(">"):
                i += 1
            body.append("%FIG2%")
            continue

        # headers
        if st.startswith("#### "):
            body.append("\\paragraph{" + inline(st[5:]) + "}"); i += 1; continue
        if st.startswith("### "):
            body.append("\\subsection{" + inline(re.sub(r'^\d+\.\d+\s+', '', st[4:])) + "}"); i += 1; continue
        if st.startswith("## "):
            name = st[3:]
            # special unnumbered / appendix handling
            if name.startswith("Appendix A"):
                body.append("\\appendix"); in_appendix = True
                body.append("\\section{Judge prompts (verbatim)}"); i += 1; continue
            if name.startswith("Appendix B"):
                body.append("\\section{Deployed faithfulness judges are content-support-only}"); i += 1; continue
            if name.startswith("Appendix C"):
                body.append("\\section{Full pairwise Cohen's $\\kappa$ (adversarial pool)}"); i += 1; continue
            if name.startswith("References"):
                body.append("%REFERENCES%"); in_references = True; i += 1; continue
            if name.startswith("Data and code"):
                body.append("\\section*{Data and code availability}"); i += 1; continue
            if name.startswith("AI-usage disclosure"):
                body.append("\\section*{AI-usage disclosure}"); i += 1; continue
            name = re.sub(r"^\d+\.\s+", "", name)
            body.append("\\section{" + inline(name) + "}"); i += 1; continue

        if st == "## Abstract" or st == "Abstract":
            i += 1; continue

        # unordered list -> itemize (skip the markdown reference list, which assemble_tex
        # strips and replaces with the .bib bibliography)
        if st.startswith("- ") and not in_references:
            items = []
            cur = st[2:]
            i += 1
            while i < len(lines):
                raw = lines[i]
                cst = raw.strip()
                if cst.startswith("- "):
                    items.append(cur); cur = cst[2:]; i += 1
                elif cst and (raw.startswith(" ") or raw.startswith("\t")):
                    cur += " " + cst; i += 1          # wrapped continuation line
                else:
                    break
            items.append(cur)
            body.append("\\begin{itemize}\n" +
                        "\n".join(r"\item " + inline(it) for it in items) +
                        "\n\\end{itemize}")
            continue

        # blank
        if not st:
            body.append(""); i += 1; continue

        # normal paragraph: accumulate soft-wrapped lines so inline markdown
        # (bold labels, emphasis) that spans a line break is processed as one unit
        para = [st]
        i += 1
        while i < len(lines) and not structural(lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        body.append(inline(" ".join(para)))

    Path("latex").mkdir(exist_ok=True)
    OUT.write_text("\n".join(body))
    print(f"wrote {OUT} ({len(body)} blocks)")


if __name__ == "__main__":
    main()
