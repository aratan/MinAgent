"""Tests for reading data files and writing PDFs out of them.

The fixtures are real files built here rather than checked-in binaries, so the
tests exercise the same readers a user's spreadsheet and report will go through:
a real zip-based .xlsx, a real .docx, a real PDF written by PyMuPDF and read back
out of the generated file. A round trip that only ever checks the text the tool
returned would pass while the PDF it wrote was empty.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pytest

from minagent.app import MinAgent
from minagent.capabilities import build_builtin_capabilities
from minagent.documents import (
    CREATE_PDF_TOOL_NAME,
    READ_DOCUMENT_TOOL_NAME,
    _is_date_format,
    create_document_tools,
    run_create_pdf,
    run_read_document,
)
from minagent.errors import AgentError
from minagent.markdown_html import markdown_to_html
from minagent.workspace import WorkspaceAccess

pymupdf = pytest.importorskip("pymupdf", reason="the documents extra is not installed")

_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _workspace(tmp_path: Path) -> WorkspaceAccess:
    return WorkspaceAccess(str(tmp_path), "Test", 0)


def _write_pdf(path: Path, lines: list[str]) -> None:
    document = pymupdf.open()
    for chunk in lines:
        page = document.new_page()
        page.insert_text((72, 100), chunk, fontsize=14)
    document.save(str(path))
    document.close()


def _write_docx(path: Path) -> None:
    document = """<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>Acta de la reunion</w:t></w:r></w:p>
<w:tbl>
<w:tr><w:tc><w:p><w:r><w:t>Nombre</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Cargo</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>Ana</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Jefa</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>
</w:body></w:document>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document)


def _write_xlsx(path: Path) -> None:
    shared = (
        f'<?xml version="1.0"?><sst xmlns="{_NS}">'
        "<si><t>Region</t></si><si><t>Ventas</t></si><si><t>Fecha</t></si><si><t>Norte</t></si>"
        "</sst>"
    )
    sheet = (
        f'<?xml version="1.0"?><worksheet xmlns="{_NS}"><sheetData>'
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>'
        '<row r="2"><c r="A2" t="s"><v>3</v></c><c r="B2"><v>1200</v></c><c r="C2" s="1"><v>45000</v></c></row>'
        "</sheetData></worksheet>"
    )
    workbook = (
        f'<?xml version="1.0"?><workbook xmlns="{_NS}" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Ventas" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels = (
        '<?xml version="1.0"?><Relationships '
        'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>'
    )
    styles = f'<?xml version="1.0"?><styleSheet xmlns="{_NS}"><cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="14"/></cellXfs></styleSheet>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/styles.xml", styles)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


async def test_reads_a_pdf_by_page_range(tmp_path: Path) -> None:
    _write_pdf(tmp_path / "informe.pdf", ["Portada del informe", "Segunda pagina con detalle"])
    workspace = _workspace(tmp_path)

    first = await run_read_document({"path": "informe.pdf"}, workspace)
    assert "Portada del informe" in first
    assert "Segunda pagina" in first

    second = await run_read_document({"path": "informe.pdf", "pages": "2"}, workspace)
    assert "Portada del informe" not in second
    assert "Segunda pagina con detalle" in second


async def test_refuses_a_page_outside_the_document(tmp_path: Path) -> None:
    _write_pdf(tmp_path / "corto.pdf", ["Una sola pagina"])
    with pytest.raises(AgentError, match="page"):
        await run_read_document({"path": "corto.pdf", "pages": "9"}, _workspace(tmp_path))


async def test_reads_a_csv_as_a_table_with_a_profile(tmp_path: Path) -> None:
    (tmp_path / "datos.csv").write_text(
        "region,ventas\nNorte,1200\nSur,980\nNorte,1430\n", encoding="utf-8"
    )
    result = await run_read_document({"path": "datos.csv"}, _workspace(tmp_path))

    assert "| region | ventas |" in result
    assert "3 row(s) after the header" in result
    # The profile is what lets a report be written from numbers rather than guesswork.
    assert "min 980" in result
    assert "max 1430" in result


async def test_pages_through_a_long_csv(tmp_path: Path) -> None:
    rows = "\n".join(f"Norte,{index}" for index in range(1, 101))
    (tmp_path / "largo.csv").write_text(f"region,ventas\n{rows}\n", encoding="utf-8")
    workspace = _workspace(tmp_path)

    first = await run_read_document({"path": "largo.csv", "limit": 10}, workspace)
    assert "| Norte | 1 |" in first
    assert "Re-read with offset=11" in first

    second = await run_read_document({"path": "largo.csv", "offset": 11, "limit": 10}, workspace)
    assert "| Norte | 11 |" in second
    assert "| Norte | 1 |" not in second


async def test_reads_json_records_as_a_table(tmp_path: Path) -> None:
    (tmp_path / "datos.json").write_text(
        json.dumps([{"a": 1, "b": "dos"}, {"a": 3, "b": "cuatro"}]), encoding="utf-8"
    )
    result = await run_read_document({"path": "datos.json"}, _workspace(tmp_path))

    assert "| a | b |" in result
    assert "cuatro" in result


async def test_reads_html_as_text_without_its_markup(tmp_path: Path) -> None:
    (tmp_path / "notas.html").write_text(
        "<html><head><style>p{color:red}</style></head><body><h1>Titulo</h1><p>Texto <b>negrita</b>.</p></body></html>",
        encoding="utf-8",
    )
    result = await run_read_document({"path": "notas.html"}, _workspace(tmp_path))

    assert "Titulo" in result
    assert "Texto negrita." in result
    assert "color:red" not in result


async def test_reads_a_docx_including_its_table(tmp_path: Path) -> None:
    _write_docx(tmp_path / "acta.docx")
    result = await run_read_document({"path": "acta.docx"}, _workspace(tmp_path))

    assert "Acta de la reunion" in result
    assert "| Nombre | Cargo |" in result
    assert "| Ana | Jefa |" in result


async def test_reads_an_xlsx_and_turns_its_date_format_into_a_date(tmp_path: Path) -> None:
    _write_xlsx(tmp_path / "ventas.xlsx")
    result = await run_read_document({"path": "ventas.xlsx"}, _workspace(tmp_path))

    assert "| Region | Ventas | Fecha |" in result
    assert "| Norte | 1200 |" in result
    # 45000 is the serial for 2023-03-15; leaving it raw would put a number in a report.
    assert "2023-03-15" in result


def test_a_red_negative_number_format_is_not_a_date() -> None:
    assert not _is_date_format("0.00;[Red]-0.00")
    assert _is_date_format("dd/mm/yyyy")
    assert not _is_date_format("General")


async def test_refuses_a_file_that_is_not_there(tmp_path: Path) -> None:
    with pytest.raises(AgentError):
        await run_read_document({"path": "nada.csv"}, _workspace(tmp_path))


async def test_creates_a_pdf_that_can_be_read_back(tmp_path: Path) -> None:
    content = (
        "# Informe de ventas\n\n"
        "Resumen de **enero** con *énfasis*.\n\n"
        "| Region | Ventas |\n|---|---:|\n| Norte | 1200 |\n| Sur | 980 |\n"
    )
    result = await run_create_pdf([{"name": "informe", "content": content}], str(tmp_path))
    assert "salida/informe.pdf" in result

    written = tmp_path / "salida" / "informe.pdf"
    assert written.exists()
    document = pymupdf.open(written)
    try:
        text = "\n".join(document[number].get_text() for number in range(document.page_count))
        assert document.page_count == 1
    finally:
        document.close()
    assert "Informe de ventas" in text
    # Bold has to be bold markup, not the four characters "<b>".
    assert "<b>" not in text
    assert "enero" in text
    assert "Norte" in text and "1200" in text


async def test_never_overwrites_an_earlier_pdf(tmp_path: Path) -> None:
    content = "# Uno"
    await run_create_pdf([{"name": "informe", "content": content}], str(tmp_path))
    await run_create_pdf([{"name": "informe", "content": content}], str(tmp_path))

    assert (tmp_path / "salida" / "informe.pdf").exists()
    assert (tmp_path / "salida" / "informe-2.pdf").exists()


async def test_writes_several_files_in_one_call(tmp_path: Path) -> None:
    result = await run_create_pdf(
        [
            {"name": "enero", "content": "# Enero"},
            {"name": "febrero", "content": "# Febrero"},
        ],
        str(tmp_path),
    )
    assert "salida/enero.pdf" in result
    assert "salida/febrero.pdf" in result
    assert (tmp_path / "salida" / "enero.pdf").exists()
    assert (tmp_path / "salida" / "febrero.pdf").exists()


async def test_a_traversing_name_is_reduced_to_the_output_folder(tmp_path: Path) -> None:
    """``../fuera`` cannot escape: the name is cut down to a single safe segment."""
    result = await run_create_pdf([{"name": "../../fuera", "content": "# X"}], str(tmp_path))

    assert "salida/fuera.pdf" in result
    assert (tmp_path / "salida" / "fuera.pdf").exists()
    assert not (tmp_path.parent / "fuera.pdf").exists()


async def test_paginates_a_long_table_without_losing_a_row(tmp_path: Path) -> None:
    rows = "\n".join(f"| Fila {index} | dato | {index} |" for index in range(1, 121))
    content = f"# Tabla\n\n| Id | Texto | Valor |\n|---|---|---|\n{rows}\n"
    await run_create_pdf([{"name": "tabla", "content": content}], str(tmp_path))

    document = pymupdf.open(tmp_path / "salida" / "tabla.pdf")
    try:
        assert document.page_count > 1
        text = re.sub(r"\s+", " ", " ".join(document[number].get_text() for number in range(document.page_count)))
    finally:
        document.close()
    for index in range(1, 121):
        assert f"Fila {index} " in text
    # The header repeats on the continuation pages, or page two is unreadable.
    assert text.count("Id Texto Valor") > 1


def test_markdown_tables_and_emphasis_become_html() -> None:
    html = markdown_to_html("Titulo\n=====\n\n**fuerte** y `a < b`\n\n- uno\n- dos\n")

    assert "<h1>Titulo</h1>" in html
    assert "<b>fuerte</b>" in html
    assert "&lt; b" in html
    assert html.count("<li>") == 2


def test_a_lone_rule_is_not_a_heading() -> None:
    html = markdown_to_html("Parrafo.\n\n---\n\nOtro parrafo.\n")

    assert "<hr/>" in html
    assert "<h2>" not in html


def test_escapes_markup_that_the_model_meant_as_text() -> None:
    html = markdown_to_html("El tag <b>no</b> es texto.\n")

    assert "&lt;b&gt;" in html
    assert "<b>no</b>" not in html


def test_a_javascript_link_keeps_only_its_label() -> None:
    html = markdown_to_html("[pulsa](javascript:alert(1))")

    assert "javascript" not in html
    assert "pulsa" in html


def test_both_tools_are_registered_as_one_capability() -> None:
    names = [tool["function"]["name"] for tool in create_document_tools()]
    assert names == [READ_DOCUMENT_TOOL_NAME, CREATE_PDF_TOOL_NAME]

    entries = build_builtin_capabilities(terminal_mode="off")
    documents = next(entry for entry in entries if entry.name == "documents")
    assert documents.tool_names == (READ_DOCUMENT_TOOL_NAME, CREATE_PDF_TOOL_NAME)
    # On demand like any other: an unused PDF tool costs the index line, not the window.
    assert not documents.eager


async def test_the_agent_dispatches_both_tools(tmp_path: Path) -> None:
    agent = MinAgent()
    agent.root_directory = str(tmp_path)
    agent.workspace_access = _workspace(tmp_path)
    (tmp_path / "datos.csv").write_text("region,ventas\nNorte,1200\n", encoding="utf-8")

    read = await agent.execute_tool(READ_DOCUMENT_TOOL_NAME, {"path": "datos.csv"}, annotate=False)
    assert "Norte" in read

    written = await agent.execute_tool(
        CREATE_PDF_TOOL_NAME,
        {"documents": [{"name": "desde-agente", "content": "# Desde el agente"}]},
        annotate=False,
    )
    assert "salida/desde-agente.pdf" in written
    assert (tmp_path / "salida" / "desde-agente.pdf").exists()


async def test_creating_a_pdf_counts_as_a_mutation(tmp_path: Path) -> None:
    """A turn that claims to have produced a file must have run a writing tool."""
    agent = MinAgent()
    agent.root_directory = str(tmp_path)
    agent.workspace_access = _workspace(tmp_path)
    agent.rebuild_capabilities()
    agent._mutations_this_turn.clear()

    await agent.execute_tool(
        CREATE_PDF_TOOL_NAME,
        {"documents": [{"name": "x", "content": "# X"}]},
        annotate=False,
    )
    assert CREATE_PDF_TOOL_NAME in agent._mutations_this_turn


def test_the_tools_cost_nothing_until_they_are_asked_for(tmp_path: Path) -> None:
    """On demand in and out: not in the prompt until used, gone again after."""
    agent = MinAgent()
    agent.root_directory = str(tmp_path)
    agent.rebuild_capabilities()

    assert READ_DOCUMENT_TOOL_NAME not in agent._published_tool_names()
    assert CREATE_PDF_TOOL_NAME not in agent._published_tool_names()

    agent.load_capabilities(["documents"])
    assert READ_DOCUMENT_TOOL_NAME in agent._published_tool_names()

    for _ in range(agent.capability_idle_turns):
        agent.age_capabilities()
    assert "documents" not in (agent.capabilities.loaded if agent.capabilities else set())
    assert CREATE_PDF_TOOL_NAME not in agent._published_tool_names()