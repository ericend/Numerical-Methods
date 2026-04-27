"""
Submission sanitizer: strip plotting code and fix Unicode in Python source.

- Removes # [PLOT_START] ... # [PLOT_END] blocks
- Removes matplotlib / plotly imports
- Sanitizes Unicode only inside COMMENT and STRING tokens
- Preserves all indentation and code structure
- Writes cleaned copies to Appendix/
"""

import io
import pathlib
import re
import tokenize
import unicodedata

# ── Regex patterns ─────────────────────────────────────────────────────────

PLOT_BLOCK = re.compile(
    r"^[ \t]*# \[PLOT_START\][^\n]*\n.*?^[ \t]*# \[PLOT_END\][^\n]*\n?",
    flags=re.MULTILINE | re.DOTALL,
)

PLOT_IMPORTS = re.compile(
    r"""
    ^[ \t]*
    (?:
        import[ \t]+(?:matplotlib(?:\.[\w.]+)?|plotly(?:\.[\w.]+)?)(?:[ \t]+as[ \t]+\w+)?
        |
        from[ \t]+(?:matplotlib(?:\.[\w.]+)?|plotly(?:\.[\w.]+)?)[ \t]+import[ \t]+.*
    )
    [ \t]*(?:\#.*)?\n
    """,
    flags=re.MULTILINE | re.VERBOSE,
)

EXCESS_BLANK = re.compile(r"\n{3,}")
TRAILING_WS = re.compile(r"[ \t]+$", flags=re.MULTILINE)

# ── Unicode replacement table ───────────────────────────────────────────────

REPLACEMENTS = {
    # Spaces / invisible
    "\u00a0": " ",  # non-breaking space
    "\u2000": " ",  # en quad
    "\u2001": " ",  # em quad
    "\u2002": " ",  # en space
    "\u2003": " ",  # em space
    "\u2004": " ",  # three-per-em space
    "\u2005": " ",  # four-per-em space
    "\u2006": " ",  # six-per-em space
    "\u2007": " ",  # figure space
    "\u2008": " ",  # punctuation space
    "\u2009": " ",  # thin space
    "\u200a": " ",  # hair space
    "\u200b": "",  # zero-width space
    "\u200c": "",  # zero-width non-joiner
    "\u200d": "",  # zero-width joiner
    "\u202f": " ",  # narrow no-break space
    "\u2060": "",  # word joiner
    "\ufeff": "",  # BOM / zero-width no-break space
    # Quotes / apostrophes
    "\u2018": "'",  # '
    "\u2019": "'",  # '
    "\u201a": "'",  # ‚
    "\u201b": "'",  # ‛
    "\u2032": "'",  # ′
    "\u2035": "'",  # ‵
    # Double quotes
    "\u201c": '"',  # "
    "\u201d": '"',  # "
    "\u201e": '"',  # „
    "\u201f": '"',  # ‟
    "\u2033": '"',  # ″
    "\u2036": '"',  # ‶
    # Dashes / hyphens / minus
    "\u2010": "-",  # ‐
    "\u2011": "-",  # ‑
    "\u2012": "-",  # ‒
    "\u2013": "-",  # –
    "\u2014": "-",  # —
    "\u2015": "-",  # ―
    "\u2212": "-",  # −
    # Ellipsis / bullets / misc punctuation
    "\u2026": "...",  # …
    "\u00b7": "*",  # ·
    "\u2022": "*",  # •
    "\u2043": "-",  # ⁃
    "\u2217": "*",  # ∗
    # Arrows
    "\u2190": "<-",  # ←
    "\u2192": "->",  # →
    "\u2194": "<->",  # ↔
    "\u21d0": "<=",  # ⇐
    "\u21d2": "=>",  # ⇒
    "\u21d4": "<=>",  # ⇔
    # Math operators
    "\u00d7": "x",  # ×
    "\u00f7": "/",  # ÷
    "\u00b1": "+/-",  # ±
    "\u2213": "-/+",  # ∓
    "\u2260": "!=",  # ≠
    "\u2248": "~=",  # ≈
    "\u2245": "~=",  # ≅
    "\u2264": "<=",  # ≤
    "\u2265": ">=",  # ≥
    "\u226a": "<<",  # ≪
    "\u226b": ">>",  # ≫
    "\u221d": "~",  # ∝
    "\u221e": "inf",  # ∞
    "\u221a": "sqrt",  # √
    "\u221b": "cuberoot",  # ∛
    "\u00ac": "not ",  # ¬
    "\u2227": "and",  # ∧
    "\u2228": "or",  # ∨
    "\u2208": " in ",  # ∈
    "\u2209": " notin ",  # ∉
    "\u220b": " contains ",  # ∋
    "\u2282": " subset ",  # ⊂
    "\u2286": " subseteq ",  # ⊆
    "\u2283": " superset ",  # ⊃
    "\u2287": " superseteq ",  # ⊇
    "\u2229": " intersect ",  # ∩
    "\u222a": " union ",  # ∪
    "\u2205": "emptyset",  # ∅
    # Calculus / algebra
    "\u2202": "d",  # ∂
    "\u2207": "nabla",  # ∇
    "\u222b": "int",  # ∫
    "\u222c": "iint",  # ∬
    "\u222d": "iiint",  # ∭
    "\u220f": "prod",  # ∏
    "\u2211": "sum",  # ∑
    # Units / degree
    "\u00b0": " deg",  # °
    "\u2126": "Ohm",  # Ω
    "\u00b5": "mu",  # µ
    # Greek uppercase
    "\u0391": "Alpha",  # Α
    "\u0392": "Beta",  # Β
    "\u0393": "Gamma",  # Γ
    "\u0394": "Delta",  # Δ
    "\u0398": "Theta",  # Θ
    "\u039b": "Lambda",  # Λ
    "\u039e": "Xi",  # Ξ
    "\u03a0": "Pi",  # Π
    "\u03a3": "Sigma",  # Σ
    "\u03a6": "Phi",  # Φ
    "\u03a8": "Psi",  # Ψ
    "\u03a9": "Omega",  # Ω
    # Greek lowercase
    "\u03b1": "alpha",  # α
    "\u03b2": "beta",  # β
    "\u03b3": "gamma",  # γ
    "\u03b4": "delta",  # δ
    "\u03b5": "epsilon",  # ε
    "\u03b6": "zeta",  # ζ
    "\u03b7": "eta",  # η
    "\u03b8": "theta",  # θ
    "\u03bb": "lambda",  # λ
    "\u03bc": "mu",  # μ
    "\u03bd": "nu",  # ν
    "\u03be": "xi",  # ξ
    "\u03c0": "pi",  # π
    "\u03c1": "rho",  # ρ
    "\u03c3": "sigma",  # σ
    "\u03c4": "tau",  # τ
    "\u03c6": "phi",  # φ
    "\u03c7": "chi",  # χ
    "\u03c8": "psi",  # ψ
    "\u03c9": "omega",  # ω
    # Blackboard bold
    "\u2115": "N",  # ℕ
    "\u2124": "Z",  # ℤ
    "\u211a": "Q",  # ℚ
    "\u211d": "R",  # ℝ
    "\u2102": "C",  # ℂ
    # Latin
    "\u00df": "ss",  # ß
}


# ── Core helpers ────────────────────────────────────────────────────────────


def read_safely(path: pathlib.Path) -> str:
    """Read file bytes; try common encodings before falling back."""
    data = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def _apply_replacements(text: str) -> str:
    """Apply REPLACEMENTS dict to a token value string."""
    text = unicodedata.normalize("NFC", text)
    for bad, good in REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text


def sanitize_unicode(src: str) -> str:
    """
    Apply Unicode replacements *only* inside COMMENT and STRING tokens.
    All other tokens (code, indentation, operators) are passed through unchanged.
    Falls back to plain text replacement if tokenization fails.
    """
    try:
        tokens = tokenize.tokenize(io.BytesIO(src.encode("utf-8")).readline)
        result = []
        for tok in tokens:
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                result.append(tok._replace(string=_apply_replacements(tok.string)))
            else:
                result.append(tok)
        return tokenize.untokenize(result).decode("utf-8")

    except tokenize.TokenError:
        # Source already has syntax issues (e.g. partial file) — fall back
        return _apply_replacements(src)


def strip_text(src: str) -> str:
    # 1. Remove plotting blocks and imports first (regex is safe here)
    src = PLOT_BLOCK.sub("", src)
    src = PLOT_IMPORTS.sub("", src)
    # 2. Normalize line endings before tokenizing
    src = src.replace("\r\n", "\n").replace("\r", "\n")
    # 3. Token-aware Unicode sanitization (only touches COMMENT + STRING)
    src = sanitize_unicode(src)
    # 4. Tidy whitespace
    src = TRAILING_WS.sub("", src)
    src = EXCESS_BLANK.sub("\n\n", src)
    return src.strip() + "\n"


def main() -> None:
    root = pathlib.Path(__file__).parent
    appendix_dir = root / "Appendix"
    appendix_dir.mkdir(exist_ok=True)

    py_files = [p for p in root.glob("*.py") if p.name != pathlib.Path(__file__).name]

    if not py_files:
        print("No Python files found.")
        return

    for src_path in sorted(py_files):
        raw = read_safely(src_path)
        cleaned = strip_text(raw)
        out_path = appendix_dir / src_path.name
        out_path.write_text(cleaned, encoding="utf-8", newline="\n")
        print(f"{src_path.name} -> Appendix/{src_path.name}")

    print(f"\nDone - {len(py_files)} file(s) written to {appendix_dir}.")


if __name__ == "__main__":
    main()
