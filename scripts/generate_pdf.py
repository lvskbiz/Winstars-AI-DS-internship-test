from __future__ import annotations

import re
from pathlib import Path


PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT_MARGIN = 54
RIGHT_MARGIN = 54
TOP_MARGIN = 64
BOTTOM_MARGIN = 64
FONT_SIZE = 11
LEADING = 15
MAX_CHARS = 88


def normalize_markdown(text: str) -> list[str]:
    lines: list[str] = []
    in_code_block = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue

        if not line:
            lines.append("")
            continue

        if in_code_block:
            lines.append("    " + line)
            continue

        if line.startswith("# "):
            lines.append(line[2:].strip().upper())
            lines.append("")
            continue

        if line.startswith("## "):
            lines.append(line[3:].strip())
            lines.append("")
            continue

        if line.startswith("### "):
            lines.append(line[4:].strip())
            continue

        if line.startswith("#### "):
            lines.append(line[5:].strip())
            continue

        if set(line) == {"-"} and len(line) >= 3:
            lines.append("")
            continue

        lines.append(line)

    return lines


def wrap_line(line: str, width: int = MAX_CHARS) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return [""]

    bullet_match = re.match(r"^([-*]|\d+\.)\s+(.*)$", stripped)
    if bullet_match:
        prefix = bullet_match.group(1) + " "
        content = bullet_match.group(2)
        words = content.split()
        lines = []
        current = prefix
        for word in words:
            candidate = word if current == prefix else current + " " + word
            if len(candidate) <= width:
                current = candidate
            else:
                lines.append(current)
                current = "  " + word
        lines.append(current)
        return lines

    words = stripped.split()
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def paginate(lines: list[str]) -> list[list[str]]:
    max_lines = int((PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN) / LEADING)
    pages: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        wrapped = wrap_line(line)
        if len(current) + len(wrapped) > max_lines:
            pages.append(current)
            current = []
        current.extend(wrapped)

    if current:
        pages.append(current)

    return pages


def build_content_stream(page_lines: list[str], page_number: int, page_count: int) -> bytes:
    y = PAGE_HEIGHT - TOP_MARGIN
    parts = ["BT", f"/F1 {FONT_SIZE} Tf", f"{LEADING} TL", f"1 0 0 1 {LEFT_MARGIN} {y} Tm"]

    for line in page_lines:
        text = escape_pdf_text(line)
        parts.append(f"({text}) Tj")
        parts.append("T*")

    footer = f"Page {page_number} of {page_count}"
    footer_y = BOTTOM_MARGIN / 2
    parts.extend(
        [
            "ET",
            "BT",
            f"/F1 10 Tf",
            f"1 0 0 1 {PAGE_WIDTH - RIGHT_MARGIN - 70} {footer_y} Tm",
            f"({escape_pdf_text(footer)}) Tj",
            "ET",
        ]
    )

    return "\n".join(parts).encode("latin-1", errors="replace")


def write_pdf(pages: list[list[str]], output_path: Path) -> None:
    objects: list[bytes] = []

    def add_object(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages_id = add_object(b"<< /Type /Pages /Kids [] /Count 0 >>")

    page_ids: list[int] = []

    for index, page in enumerate(pages, start=1):
        content_stream = build_content_stream(page, index, len(pages))
        content_obj = add_object(
            f"<< /Length {len(content_stream)} >>\nstream\n".encode("latin-1")
            + content_stream
            + b"\nendstream"
        )
        page_obj = add_object(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_obj} 0 R >>"
            ).encode("latin-1")
        )
        page_ids.append(page_obj)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]

    for object_number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("latin-1")
    )

    output_path.write_bytes(pdf)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    source_path = project_root / "docs" / "code_structure_guide.md"
    output_path = project_root / "docs" / "code_structure_guide.pdf"

    raw_text = source_path.read_text(encoding="utf-8")
    normalized_lines = normalize_markdown(raw_text)
    pages = paginate(normalized_lines)
    write_pdf(pages, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
