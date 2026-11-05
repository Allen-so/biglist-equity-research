"""Universe container for a collection of symbols or rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List

import pandas as pd


@dataclass
class Universe:
    data: pd.DataFrame
    name: str = 'universe'

    @classmethod
    def from_frame(cls, data: pd.DataFrame, name: str = 'universe') -> 'Universe':
        return cls(data=data.copy(), name=name)

    @classmethod
    def from_csv(cls, path: str | Path, name: str = 'universe') -> 'Universe':
        return cls(data=pd.read_csv(path), name=name)

    def copy(self) -> 'Universe':
        return Universe(data=self.data.copy(), name=self.name)

    def as_dataframe(self) -> pd.DataFrame:
        return self.data.copy()

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[pd.Series]:
        return iter(self.data.itertuples(index=False, name=None))

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.data.head(n)

    @property
    def symbols(self) -> List[Any]:
        if 'symbol' in self.data.columns:
            return list(self.data['symbol'].tolist())
        return list(range(len(self.data)))

    def filter(self, predicate, *args, **kwargs) -> 'Universe':
        if callable(predicate):
            filtered = self.data[predicate(self.data, *args, **kwargs)]
        else:
            filtered = self.data
        return Universe(data=filtered, name=self.name)

    def add_column(self, name: str, values: Iterable[Any]) -> None:
        self.data[name] = list(values)
