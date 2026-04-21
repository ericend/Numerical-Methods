"""
Strip plotting code from Python files.

This script processes Python files in the current directory, removing:
- Code blocks marked with # [PLOT_START] and # [PLOT_END] comments
- Imports of matplotlib and plotly libraries
- Excess blank lines (more than 2 consecutive newlines)

The cleaned files are saved to an 'Appendix' subdirectory, useful for
creating versions of scripts without plotting dependencies for submission
or archival purposes.
"""

import pathlib
import re

PLOT_BLOCK = re.compile(
    r"[ \t]*# \[PLOT_START\].*?# \[PLOT_END\]\n?",
    flags=re.DOTALL,
)
PLOT_IMPORTS = re.compile(
    r"^import (matplotlib|plotly).*\n",
    flags=re.MULTILINE,
)
EXCESS_BLANK = re.compile(r"\n{3,}")


def strip(src: str) -> str:
    src = PLOT_BLOCK.sub("", src)
    src = PLOT_IMPORTS.sub("", src)
    src = EXCESS_BLANK.sub("\n\n", src)
    return src.strip() + "\n"


def main() -> None:
    root = pathlib.Path(__file__).parent  # folder the script lives in
    appendix_dir = root / "Appendix"
    appendix_dir.mkdir(exist_ok=True)  # create if missing

    py_files = [
        p
        for p in root.glob("*.py")
        if p.name != pathlib.Path(__file__).name  # skip strip_plots.py itself
    ]

    if not py_files:
        print("No Python files found.")
        return

    for src_path in sorted(py_files):
        stripped = strip(src_path.read_text(encoding="utf-8"))
        out_path = appendix_dir / src_path.name
        out_path.write_text(stripped, encoding="utf-8")
        print(f"  {src_path.name}  →  Appendix/{src_path.name}")

    print(f"\nDone — {len(py_files)} file(s) written to {appendix_dir}.")


if __name__ == "__main__":
    main()
