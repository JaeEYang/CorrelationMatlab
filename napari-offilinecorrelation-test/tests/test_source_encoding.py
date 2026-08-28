from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

# Non-ASCII text is safe in Qt widget labels but not in anything that reaches
# stdout: Windows consoles use cp1252, where printing a character outside that
# encoding raises UnicodeEncodeError. Lines that are genuinely GUI-only opt out
# with this marker, either on the line itself or on the line directly above it.
ASCII_EXEMPT_MARKER = "ascii-exempt"


def _is_exempt(lines: list[str], index: int) -> bool:
    if ASCII_EXEMPT_MARKER in lines[index]:
        return True
    return index > 0 and ASCII_EXEMPT_MARKER in lines[index - 1]


def _escaped(line: str) -> str:
    """Render an offending line so the failure message is itself printable.

    Reporting the raw line would put the offending character straight back onto
    the cp1252 stdout this test exists to protect, turning a clear failure into
    a UnicodeEncodeError. Escaping also names the exact codepoint at fault.
    """
    return line.strip().encode("ascii", "backslashreplace").decode("ascii")


def test_source_files_are_ascii_outside_marked_lines():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        offenders.extend(
            f"{path.relative_to(SRC).as_posix()}:{index + 1}: {_escaped(line)}"
            for index, line in enumerate(lines)
            if not line.isascii() and not _is_exempt(lines, index)
        )

    assert not offenders, (
        "Non-ASCII characters outside lines marked "
        f"'{ASCII_EXEMPT_MARKER}':\n" + "\n".join(offenders)
    )
