"""Table serialization and chunking utilities preserving 2D topology and headers."""

from __future__ import annotations

import re
from typing import Sequence
import tiktoken


class TableFormatter:
    """Formats 2D table data into GitHub-Flavored Markdown tables and splits large tables
    while preserving header rows across all chunks.
    """

    def __init__(self, tokenizer_name: str = "cl100k_base") -> None:
        self._encoding = tiktoken.get_encoding(tokenizer_name)

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    @staticmethod
    def _clean_cell(cell: str | None) -> str:
        if cell is None:
            return ""
        # Replace newlines within a cell with HTML break so table markdown doesn't break
        cleaned = re.sub(r"[\r\n]+", "<br>", str(cell).strip())
        # Replace pipes to avoid breaking table column delimiters
        return cleaned.replace("|", "&#124;")

    def to_markdown(self, rows: Sequence[Sequence[str | None]], caption: str | None = None) -> str:
        """Converts a 2D sequence of cells into a GitHub-Flavored Markdown table."""
        if not rows or not rows[0]:
            return ""

        clean_rows = [[self._clean_cell(c) for c in row] for row in rows]
        header = clean_rows[0]
        data_rows = clean_rows[1:]

        col_widths = [len(col) for col in header]
        for row in data_rows:
            for idx, cell in enumerate(row):
                if idx < len(col_widths):
                    col_widths[idx] = max(col_widths[idx], len(cell))
                else:
                    col_widths.append(len(cell))

        # Pad columns
        header_line = "| " + " | ".join(header[i].ljust(col_widths[i]) for i in range(len(header))) + " |"
        sep_line = "| " + " | ".join("-" * max(3, col_widths[i]) for i in range(len(header))) + " |"

        lines: list[str] = []
        if caption:
            lines.append(f"**Table: {caption.strip()}**\n")

        lines.append(header_line)
        lines.append(sep_line)

        for row in data_rows:
            # Pad row if shorter than header
            padded_row = row + [""] * max(0, len(header) - len(row))
            row_line = "| " + " | ".join(padded_row[i].ljust(col_widths[i]) for i in range(len(header))) + " |"
            lines.append(row_line)

        return "\n".join(lines)

    def chunk_table(
        self,
        rows: Sequence[Sequence[str | None]],
        max_tokens: int = 500,
        caption: str | None = None,
    ) -> list[str]:
        """Splits a large table into chunks while repeating the header row on every chunk."""
        if not rows or len(rows) < 2:
            return [self.to_markdown(rows, caption=caption)] if rows else []

        full_md = self.to_markdown(rows, caption=caption)
        if self.count_tokens(full_md) <= max_tokens:
            return [full_md]

        header = rows[0]
        data_rows = rows[1:]
        chunks: list[str] = []
        current_subrows: list[Sequence[str | None]] = []
        start_row_idx = 1

        for i, row in enumerate(data_rows, start=1):
            test_subrows = current_subrows + [row]
            test_table = [header] + test_subrows
            test_md = self.to_markdown(test_table, caption=f"{caption} (Rows {start_row_idx}-{i})" if caption else None)

            if self.count_tokens(test_md) > max_tokens and current_subrows:
                # Flush current subtable
                flush_table = [header] + current_subrows
                end_idx = start_row_idx + len(current_subrows) - 1
                sub_cap = f"{caption} (Rows {start_row_idx}-{end_idx})" if caption else f"Rows {start_row_idx}-{end_idx}"
                chunks.append(self.to_markdown(flush_table, caption=sub_cap))

                current_subrows = [row]
                start_row_idx = i
            else:
                current_subrows.append(row)

        if current_subrows:
            end_idx = start_row_idx + len(current_subrows) - 1
            sub_cap = f"{caption} (Rows {start_row_idx}-{end_idx})" if caption else f"Rows {start_row_idx}-{end_idx}"
            chunks.append(self.to_markdown([header] + current_subrows, caption=sub_cap))

        return chunks
