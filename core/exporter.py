"""
core/exporter.py — Export batch analysis results to CSV and Excel.

Both exporters take the DataFrame produced by
core.batch_analyzer.results_to_dataframe() and return raw bytes suitable for
st.download_button.  Nothing is written to disk and the AI service is never
called.  The DataFrame holds only the uploaded CSV columns plus the ai_*
result columns, so no API key or other credential can end up in an export.

PUBLIC API
──────────
    from core.exporter import to_csv_bytes, to_excel_bytes

    csv_bytes  = to_csv_bytes(merged_df)    # UTF-8-SIG (Excel-friendly)
    xlsx_bytes = to_excel_bytes(merged_df)  # sheet "Hasil Analisis"
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd

CSV_ENCODING = "utf-8-sig"
EXCEL_SHEET_NAME = "Hasil Analisis"
CSV_MIME = "text/csv"
EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize *df* to CSV bytes with a UTF-8 BOM so Excel detects the encoding."""
    return df.to_csv(index=False).encode(CSV_ENCODING)


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = EXCEL_SHEET_NAME) -> bytes:
    """
    Serialize *df* to an .xlsx workbook (single sheet) and return its bytes.

    Text that starts with "=" (e.g. a review like "=bagus") is stored as plain
    text rather than an Excel formula, so the workbook shows the original
    value and cannot execute injected formulas.
    """
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        sheet = writer.sheets[sheet_name]
        for row in sheet.iter_rows():
            for cell in row:
                if cell.data_type == "f":
                    cell.data_type = "s"
        # Reasonable column widths for readability (capped).
        for column_cells in sheet.columns:
            longest = max(len(str(c.value)) if c.value is not None else 0 for c in column_cells)
            sheet.column_dimensions[column_cells[0].column_letter].width = min(max(longest + 2, 10), 60)
    return buffer.getvalue()
