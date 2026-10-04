"""Markdown to the small HTML subset PyMuPDF can lay out.

``create_pdf`` renders with MuPDF, which understands HTML and CSS but not
Markdown. The ``markdown`` package would do this conversion, but it is not a
dependency of the project and the conversion needed here is narrow: what a
model writes when it is asked for a report. Headings, paragraphs, emphasis,
lists, tables, code, quotes and rules cover it, and anything outside that set
comes through as text rather than as raw markup, which is the failure mode
that matters - a stray ``<`` in a document must not be read as a tag.

So this is deliberately not a Markdown implementation. It renders the subset and
escapes everything else, because a report that shows ``**stars**`` is a bug the
user sees immediately, while one that mis-nests a list is a detail.
"""

from __future__ import annotations

import html as html_module
import re

# ---------------------------------------------------------------- inline

_CODE_SPAN = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_STRONG = re.compile(r"\*\*([^*]+)\*\*")
_STRONG_UNDERSCORE = re.compile(r"__([^_]+)__")
_EMPHASIS = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)")
_EMPHASIS_UNDERSCORE = re.compile(r"(?<![_\w])_([^_\n]+)_(?!\w)")

# ----------------------------------------------------------------- blocks

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SETEXT = re.compile(r"^\s*(=+|-{2,})\s*$")
_RULE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_ORDERED = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_QUOTE = re.compile(r"^\s*>\s?(.*)$")
_FENCE = re.compile(r"^\s*(```+|~~~+)\s*([\w+-]*)\s*$")


def _escape(text: str) -> str:
    return html_module.escape(text, quote=False)


def _inline(text: str) -> str:
    """Render the inline span markup, escaping whatever is left over."""
    placeholders: list[str] = []

    def stash(rendered: str) -> str:
        placeholders.append(rendered)
        return f"\x00{len(placeholders) - 1}\x00"

    def on_code(match: re.Match[str]) -> str:
        return stash(f"<code>{_escape(match.group(1))}</code>")

    def on_link(match: re.Match[str]) -> str:
        href = match.group(2).strip()
        # A javascript: or data: URL is markup the report must not carry.
        if not re.match(r"^(https?:|mailto:)", href, re.IGNORECASE):
            return stash(_escape(match.group(1)))
        return stash(f'<a href="{html_module.escape(href, quote=True)}">{_escape(match.group(1))}</a>')

    text = _CODE_SPAN.sub(on_code, text)
    text = _LINK.sub(on_link, text)
    # Escaping happens before the emphasis tags go in, or it escapes the tags
    # it just added and the report prints "<b>" to the reader.
    text = _escape(text)
    text = _STRONG.sub(r"<b>\1</b>", text)
    text = _STRONG_UNDERSCORE.sub(r"<b>\1</b>", text)
    text = _EMPHASIS.sub(r"<i>\1</i>", text)
    text = _EMPHASIS_UNDERSCORE.sub(r"<i>\1</i>", text)
    for index, rendered in enumerate(placeholders):
        text = text.replace(f"\x00{index}\x00", rendered)
    return text


def _split_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [cell.strip() for cell in cells]


def _table_rows(lines: list[str]) -> list[list[str]]:
    return [_split_row(line) for line in lines if not _TABLE_DIVIDER.match(line)]


def _render_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    head, *body = rows
    width = max(len(row) for row in rows)
    out = ["<table>"]
    out.append("<tr>" + "".join(f"<th>{_inline(cell)}</th>" for cell in _pad(head, width)) + "</tr>")
    for row in body:
        out.append("<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in _pad(row, width)) + "</tr>")
    out.append("</table>")
    return "".join(out)


def _pad(row: list[str], width: int) -> list[str]:
    return row + [""] * (width - len(row))


def markdown_to_html(text: str) -> str:
    """Render ``text`` as the block-level HTML the PDF renderer accepts."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    paragraph: list[str] = []
    index = 0
    total = len(lines)

    def flush_paragraph() -> None:
        if paragraph:
            joined = " ".join(line.strip() for line in paragraph).strip()
            if joined:
                out.append(f"<p>{_inline(joined)}</p>")
            paragraph.clear()

    while index < total:
        line = lines[index]

        fence = _FENCE.match(line)
        if fence:
            flush_paragraph()
            marker = fence.group(1)[0] * 3
            index += 1
            code: list[str] = []
            while index < total and not lines[index].strip().startswith(marker):
                code.append(lines[index])
                index += 1
            index += 1
            out.append("<pre>" + _escape("\n".join(code)) + "</pre>")
            continue

        if not line.strip():
            flush_paragraph()
            index += 1
            continue

        if _RULE.match(line):
            flush_paragraph()
            out.append("<hr/>")
            index += 1
            continue

        heading = _HEADING.match(line)
        # The underline form of a heading: a line of text, then a row of '=' or
        # '-'. It is checked before the rule, because '---' under a paragraph is
        # a heading and '---' on its own is a rule.
        if (
            not heading
            and line.strip()
            and not _RULE.match(line)
            and index + 1 < total
            and _SETEXT.match(lines[index + 1])
            and not _BULLET.match(line)
            and not _ORDERED.match(line)
            and not _QUOTE.match(line)
        ):
            flush_paragraph()
            level = 1 if lines[index + 1].strip().startswith("=") else 2
            out.append(f"<h{level}>{_inline(line.strip())}</h{level}>")
            index += 2
            continue

        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2).strip())}</h{level}>")
            index += 1
            continue

        if "|" in line and index + 1 < total and _TABLE_DIVIDER.match(lines[index + 1]):
            flush_paragraph()
            block = [line, lines[index + 1]]
            index += 2
            while index < total and "|" in lines[index] and lines[index].strip():
                block.append(lines[index])
                index += 1
            out.append(_render_table(_table_rows(block)))
            continue

        quote = _QUOTE.match(line)
        if quote:
            flush_paragraph()
            block = []
            while index < total and (quoted := _QUOTE.match(lines[index])):
                block.append(quoted.group(1))
                index += 1
            out.append(f"<blockquote><p>{_inline(' '.join(block))}</p></blockquote>")
            continue

        bullet = _BULLET.match(line)
        ordered = _ORDERED.match(line)
        if bullet or ordered:
            flush_paragraph()
            tag = "ul" if bullet else "ol"
            block = []
            while index < total:
                current = lines[index]
                # The list ends when the marker type changes: a bullet list
                # followed by "1." is two lists, not one list of five items.
                match = _BULLET.match(current) if tag == "ul" else _ORDERED.match(current)
                if match:
                    block.append(_inline(match.group(match.lastindex or 2)))
                    index += 1
                    continue
                other = _ORDERED.match(current) if tag == "ul" else _BULLET.match(current)
                if other:
                    break
                # A plain line under a list item continues it, as Markdown reads.
                if current.strip() and block and not _HEADING.match(current) and current.startswith((" ", "\t")):
                    block[-1] += " " + _inline(current.strip())
                    index += 1
                    continue
                break
            items = "".join(f"<li>{item}</li>" for item in block)
            out.append(f"<{tag}>{items}</{tag}>")
            continue

        paragraph.append(line)
        index += 1

    flush_paragraph()
    return "\n".join(out)