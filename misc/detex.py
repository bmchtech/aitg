#
# DETEX
# based on detex.py: https://web.archive.org/web/20200221173746/https://www.gilles-bertrand.com/2012/11/a-simple-detex-function-in-python.html
# a script to transform latex text into simple text
#

import re

def detex(latex_text, destroy_latex=False):
    """Transform a latex text into a simple text"""

    def apply_regexps(text, regexp_list, destroy):
        """Applies successively many regexps to a text"""
        # apply all the rules in the ruleset
        for element in regexp_list:
            left = element["left"]
            right = element["right"]
            if destroy:
                right = " "
            r = re.compile(left)
            text = r.sub(right, text)
        return text

    # initialization
    regexps = []
    text = latex_text
    # remove all the contents of the header, ie everything before the first occurence of "\begin{document}"
    text = re.sub(r"(?s).*?(\\begin\{document\})", "", text, 1)

    # remove comments
    regexps.append({r"left": r"([^\\])%.*", "right": r"\1"})
    text = apply_regexps(text, regexps, destroy_latex)
    regexps = []

    # - replace some LaTeX commands by the contents inside curly rackets
    to_reduce = [
        r"\\emph",
        r"\\textbf",
        r"\\textit",
        r"\\text",
        r"\\IEEEauthorblockA",
        r"\\IEEEauthorblockN",
        r"\\author",
        r"\\caption",
        r"\\author",
        r"\\thanks",
    ]
    for tag in to_reduce:
        regexps.append({"left": tag + r"\{([^\}\{]*)\}", "right": r"\1"})
    text = apply_regexps(text, regexps, destroy_latex)
    regexps = []

    # highlight

    # - replace some LaTeX commands by the contents inside curly brackets and highlight these contents
    to_highlight = [
        r"\\part[\*]*",
        r"\\chapter[\*]*",
        r"\\section[\*]*",
        r"\\subsection[\*]*",
        r"\\subsubsection[\*]*",
        r"\\paragraph[\*]*",
    ]
    # highlightment pattern: #--content--#
    for tag in to_highlight:
        regexps.append({"left": tag + r"\{([^\}\{]*)\}", "right": r"\n#--\1--#\n"})
    # highlightment pattern: [content]
    to_highlight = [r"\\title", r"\\author", r"\\thanks", r"\\cite", r"\\ref"]
    for tag in to_highlight:
        regexps.append({"left": tag + r"\{([^\}\{]*)\}", "right": r"[\1]"})
    text = apply_regexps(text, regexps, destroy_latex)
    regexps = []

    # remove LaTeX tags
    # - remove completely some LaTeX commands that take arguments
    to_remove = [
        r"\\maketitle",
        r"\\footnote",
        r"\\centering",
        r"\\IEEEpeerreviewmaketitle",
        r"\\includegraphics",
        r"\\IEEEauthorrefmark",
        r"\\label",
        r"\\begin",
        r"\\end",
        r"\\big",
        r"\\right",
        r"\\left",
        r"\\documentclass",
        r"\\usepackage",
        r"\\bibliographystyle",
        r"\\bibliography",
        r"\\cline",
        r"\\multicolumn",
    ]

    # replace tag with options and argument by a single space
    for tag in to_remove:
        regexps.append({"left": tag + r"(\[[^\]]*\])*(\{[^\}\{]*\})*", "right": r" "})
        # regexps.append({'left':tag+r'\{[^\}\{]*\}\[[^\]\[]*\]', 'right':r' '})
    text = apply_regexps(text, regexps, destroy_latex)
    regexps = []

    # - replace some LaTeX commands by the contents inside curly rackets
    # replace some symbols by their ascii equivalent
    # - common symbols
    regexps.append({"left": r"\\eg(\{\})* *", "right": r"e.g., "})
    regexps.append({"left": r"\\ldots", "right": r"..."})
    regexps.append({"left": r"\\Rightarrow", "right": r"=>"})
    regexps.append({"left": r"\\rightarrow", "right": r"->"})
    regexps.append({"left": r"\\le", "right": r"<="})
    regexps.append({"left": r"\\ge", "right": r">"})
    regexps.append({"left": r"\\_", "right": r"_"})
    regexps.append({"left": r"\\\\", "right": r"\n"})
    regexps.append({"left": r"~", "right": r" "})
    regexps.append({"left": r"\\&", "right": r"&"})
    regexps.append({"left": r"\\%", "right": r"%"})
    regexps.append({"left": r"([^\\])&", "right": r"\1\t"})
    regexps.append({"left": r"\\item", "right": r"\t- "})
    regexps.append(
        {
            "left": r"\\hline[ \t]*\\hline",
            "right": r"=============================================",
        }
    )
    regexps.append(
        {
            "left": r"[ \t]*\\hline",
            "right": r"_____________________________________________",
        }
    )
    # - special letters
    regexps.append({"left": r"\\\'{?\{e\}}?", "right": r"é"})
    regexps.append({"left": r"\\`{?\{a\}}?", "right": r"à"})
    regexps.append({"left": r"\\\'{?\{o\}}?", "right": r"ó"})
    regexps.append({"left": r"\\\'{?\{a\}}?", "right": r"á"})
    # keep untouched the contents of the equations
    regexps.append({"left": r"\$(.)\$", "right": r"\1"})
    regexps.append({"left": r"\$([^\$]*)\$", "right": r"\1"})
    # remove the equation symbols ($)
    regexps.append({"left": r"([^\\])\$", "right": r"\1"})
    # correct spacing problems
    regexps.append({"left": r" +,", "right": r","})
    regexps.append({"left": r" +", "right": r" "})
    regexps.append({"left": r" +\)", "right": r"\)"})
    regexps.append({"left": r"\( +", "right": r"\("})
    regexps.append({"left": r" +\.", "right": r"\."})
    # remove lonely curly brackets
    regexps.append({"left": r"^([^\{]*)\}", "right": r"\1"})
    regexps.append({"left": r"([^\\])\{([^\}]*)\}", "right": r"\1\2"})
    regexps.append({"left": r"\\\{", "right": r"\{"})
    regexps.append({"left": r"\\\}", "right": r"\}"})
    # strip white space characters at end of line
    regexps.append({"left": r"[ \t]*\n", "right": r"\n"})
    # remove consecutive blank lines
    regexps.append({"left": r"([ \t]*\n){3,}", "right": r"\n"})
    # apply all those regexps
    text = apply_regexps(text, regexps, destroy_latex)
    regexps = []

    # clean space
    if destroy_latex:
        text = re.sub(r"\s\s+", " ", text)
        text = text.strip()

    # return the modified text
    return text


def main():
    """Just for debugging"""

    import sys

    should_destroy = sys.argv[1] == 'destroy'

    # print "defining the test text\n"
    latex_text = r"""
    % This paper can be formatted using the peerreviewca
    % (instead of conference) mode.
    \documentclass[twocolumn,a4paper]{article}
    %\documentclass[peerreviewca]{IEEEtran}
    % correct bad hyphenation here
    \hyphenation{op-ti-cal net-works semi-con-duc-tor IEEEtran pri-va-cy Au-tho-ri-za-tion}
    % package for printing the date and time (version)
    \usepackage{time}
    \begin{document}
    \title{Next Generation Networks}
    \author{Tot titi\thanks{Network and Security -- test company -- toto@ieee.org}}
    \maketitle
    \begin{abstract}\footnote{Version :  \today ;  \now}
    lorem ipsum(\ldots)\end{abstract}
    \emph{Keywords: IP Multimedia Subsystem, Quality of Service}
    \section{Introduction} \label{sect:introduction}
    lorem ipsum(\ldots) \% of the world population. \cite{TISPAN2006a}. \footnote{Bearer Independent Call Control protocol}. 
    \hline
    \section{Protocols used in IMS} \label{sect:protocols}
    lorem ipsum(\ldots) \cite{rfc2327, rfc3264}.
    \subsection{Authentication, Authorization, and Accounting} \label{sect:protocols_aaa}
    lorem ipsum(\ldots)
    \subsubsection{Additional protocols} \label{sect:protocols_additional}
    lorem ipsum(\ldots)
    \begin{table}
        \begin{center}
            \begin{tabular}{|c|c|c|}
            \hline
                \textbf{Capability}                                 & \textbf{UE} & \textbf{GGSN} \\ \hline
                \emph{DiffServ Edge Function}           & Optional      & Required          \\ \hline
                \emph{RSVP/IntServ}                                 & Optional      & Optional          \\ \hline
                \emph{IP Policy Enforcement Point}  & Optional      & Required          \\ \hline
            \end{tabular}
        \caption{IP Bearer Services Manager capability in the UE and GGSN}
        \label{tab_ue_ggsn}
        \end{center}
    \end{table}
     The main transport layer functions are listed below:
    \begin{my_itemize}
        \item The \emph{Resource Control Enforcement Function} (RCEF) enforces policies under the control of the A-RACF. It opens and closes unidirectional filters called \emph{gates} or \emph{pinholes}, polices traffic and marks IP packets \cite{TISPAN2006c}.
        \item  The \emph{Border Gateway Function} (BGF) performs policy enforcement and Network Address Translation (NAT) functions under the control of the S-PDF. It operates on unidirectional flows related to a particular session (micro-flows) \cite{TISPAN2006c}.
        \item  The \emph{Layer 2 Termination Point} (L2TP) terminates the Layer 2 procedures of the access network \cite{TISPAN2006c}.
    \end{my_itemize}
    Their QoS capabilities are summarized in table \ref{tab_rcef_bgf} \cite{TISPAN2006c}.
    The admission control usually follows a three step procedure:
    \begin{my_enumerate}
        \item Authorization of resources (\eg by the A-RACF)
        \item Resource reservation (\eg by the BGF)
        \item Resource commitment (\eg by the RCEF)
    \end{my_enumerate}
    \begin{figure}
    \centering
    \includegraphics[width=1.5in]{./pictures/RACS_functional_architecture}
    \caption{RACS interaction with transfer functions}
    \label{fig_RACS_functional_architecture}
    \end{figure}
    %\subsection{Example}  \label{sect:qos_example}
    % conference papers do not normally have an appendix
    % use section* for acknowledgement
    \section*{Acknowledgment}
    % optional entry into table of contents (if used)
    %\addcontentsline{toc}{section}{Acknowledgment}
    lorem ipsum(\ldots)
    \bibliographystyle{plain}
    %\bibliographystyle{alpha}
    \bibliography{./mabiblio}
    \end{document}
    """
    # print '\n'.join(diff)
    text = detex(latex_text, should_destroy)

    print(text)


if __name__ == "__main__":
    main()
