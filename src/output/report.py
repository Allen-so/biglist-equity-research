"""Simple report generation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd


class ReportBuilder:
    def __init__(self, title: str = 'BigList Report') -> None:
        self.title = title

    def build(self, rows: Iterable[Mapping[str, Any]]) -> str:
        frame = pd.DataFrame(rows)
        if frame.empty:
            return f'{self.title}\nNo rows available.'
        return f'{self.title}\n{frame.to_string(index=False)}'

    def __call__(self, rows: Iterable[Mapping[str, Any]]) -> str:
        return self.build(rows)


def render_report(rows: Iterable[Mapping[str, Any]], title: str = 'BigList Report') -> str:
    return ReportBuilder(title=title).build(rows)


def generate_report(rows: Iterable[Mapping[str, Any]], title: str = 'BigList Report') -> str:
    return render_report(rows, title=title)
