"""Assemble latex/main.tex into a complete ACL document: preamble, title, abstract,
figures, §7 -> Conclusion + unnumbered Limitations/Ethics, bibliography.
"""
from __future__ import annotations
import re
from pathlib import Path

P = Path("latex/main.tex")
body = P.read_text()

# --- extract abstract (between \section{Abstract} and \section{Introduction}) ---
m = re.search(r"\\section\{Abstract\}(.*?)(\\section\{Introduction\})", body, re.DOTALL)
abstract = m.group(1).strip()
body = "\\section{Introduction}" + body.split("\\section{Introduction}", 1)[1]

# --- split §7 (Limitations, Ethics, Conclusion) ---
m = re.search(r"\\section\{Limitations, Ethics, and Conclusion\}(.*?)(\\section\*\{Data and code)",
              body, re.DOTALL)
sec7 = m.group(1)
def grab(label, text):
    mm = re.search(r"\\textbf\{" + label + r"[^}]*\}(.*?)(?=\\textbf\{|\Z)", text, re.DOTALL)
    return mm.group(1).strip() if mm else ""
lim = grab("Limitations", sec7)
eth = grab("Ethics", sec7)
con = grab("Conclusion", sec7)
sec7_new = ("\\section{Conclusion}\n" + con +
            "\n\n\\section*{Limitations}\n" + lim +
            "\n\n\\section*{Ethics Statement}\n" + eth + "\n\n")
body = body[:m.start()] + sec7_new + body[m.start()+len(m.group(0))-len("\\section*{Data and code"):]

# --- figures ---
fig2 = r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig2_attribution.pdf}
\caption{Attribution $2\times2$ on Claude-inserted citations: faithful-rate by prompt
(content vs.\ attribution) and citation correctness (correct vs.\ misattributed), for the
three LLM judges. Under the content prompt the misattributed bar sits well below the correct
bar but still near half (the judge passes most correct citations yet calls about half of
misattributed answers faithful); under the attribution prompt the misattributed bar collapses
to $\approx 3$--$4\%$ while the correct bar stays high. Significance in Table~\ref{tab:4}.}
\label{fig:2x2}
\end{figure}"""
body = body.replace("%FIG2%", fig2)

fig1 = r"""\begin{figure}[t]
\centering\small
\fbox{\parbox{0.93\columnwidth}{
\textbf{P1:} ``\ldots{}the three R's: reduce, reuse, recycle\ldots{}''\quad
\textbf{P2:} ``The 3R Initiative \ldots{} build a sound-material-cycle society.''\\[2pt]
\textbf{Gold:} The three R's are Reduce, Reuse, Recycle~[1]. The principles build a
sound-material-cycle society~[2]. {\footnotesize($[1]\!\to\!$P1 \checkmark,\,
$[2]\!\to\!$P2 \checkmark)}\\[2pt]
\texttt{citation\_relocation}: \ldots{}Recycle~[2]. \ldots{}society~[1].
{\footnotesize(each claim now mis-cited \ding{55})}\\[2pt]
Content-support judge: two supported claims $\to$ \textbf{faithful}. Attribution-aware
judge: ``three R's'' vs.\ [2] unsupported $\to$ \textbf{unfaithful}.
}}
\caption{Worked example of \texttt{citation\_relocation} (real \textsc{ExpertQA} cell,
passages abbreviated). Content is unchanged; only the \texttt{[N]} markers are deranged, so
every claim stays supported by some passage but is cited to a non-supporting one.}
\label{fig:worked}
\end{figure}"""
body = body.replace("%FIG1%", fig1)

# --- references ---
body = body.replace("%REFERENCES%", "\\bibliography{references}")
# drop the markdown reference list (lines starting with "- `key`") -> we use .bib
body = re.sub(r"\n- \\texttt\{\\detokenize\{[^}]*\}\}[^\n]*", "", body)
body = re.sub(r"\n- `[^\n]*", "", body)
# ACL order: References precede the Appendix. The markdown had appendices first, so move
# \bibliography to just before \appendix.
body = body.replace("\\bibliography{references}", "")
body = body.replace("\\appendix", "\\bibliography{references}\n\n\\FloatBarrier\n\\appendix", 1)

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[review]{acl}
\usepackage{times}
\usepackage{latexsym}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{inconsolata}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{pifont}
\usepackage{graphicx}
\usepackage{float}
\usepackage{placeins}
\usepackage{listings}
\usepackage{enumitem}
\setlist[itemize]{nosep,leftmargin=1.4em}
\lstset{basicstyle=\footnotesize\ttfamily,breaklines=true,breakatwhitespace=true,columns=fullflexible,frame=none,xleftmargin=0pt,aboveskip=4pt,belowskip=4pt}
\graphicspath{{figures/}}
\setlength{\abovecaptionskip}{4pt}
\setlength{\belowcaptionskip}{1pt}
\raggedbottom
\setlength{\textfloatsep}{8pt}
\setlength{\floatsep}{6pt}
\setlength{\intextsep}{6pt}
\title{Citation Attribution is a Blind Spot for RAG Faithfulness Judges:\\ An Audit of Seven Evaluators}
\author{Anonymous ACL submission}
\begin{document}
\maketitle
\begin{abstract}
""" + abstract + r"""
\end{abstract}

"""

final = PREAMBLE + body.strip() + "\n\n\\end{document}\n"
P.write_text(final)
print(f"assembled {P}: {len(final)} chars")
print("checks:", "abstract" in final[:1500], "| FIG markers left:", final.count("%FIG"),
      "| \\end{document}" in final)
