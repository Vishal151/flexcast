"""The model interface and storage round-trip.

A model is saved as a directory of files: whatever the model needs (e.g. LightGBM text dumps)
plus metadata.json naming the model class. No pickles: loading never executes stored code.
"""

import json
from typing import Any, ClassVar, Protocol, Self

import numpy as np
import pandas as pd

from flexcast.storage import Storage

QUANTILES = (0.1, 0.5, 0.9)
COLUMNS = ["p10", "p50", "p90"]


class Model(Protocol):
    name: ClassVar[str]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Columns p10, p50, p90 (£/MWh), index = X.index, no NaN, p10 <= p50 <= p90."""
        ...

    def to_files(self) -> dict[str, str]: ...

    @classmethod
    def from_files(cls, files: dict[str, str]) -> Self: ...


def pinball_loss(y: pd.Series, pred: pd.DataFrame) -> float:
    """Mean pinball (quantile) loss over p10/p50/p90."""
    losses = []
    for q, col in zip(QUANTILES, COLUMNS, strict=True):
        err = y.to_numpy() - pred[col].to_numpy()
        losses.append(np.maximum(q * err, (q - 1) * err).mean())
    return float(np.mean(losses))


def save_model(storage: Storage, model: Model, prefix: str, metadata: dict[str, Any]) -> None:
    files = model.to_files()
    for name, text in files.items():
        storage.write_text(text, f"{prefix}/{name}")
    meta = {**metadata, "model": model.name, "files": sorted(files)}
    storage.write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", f"{prefix}/metadata.json"
    )


def load_model(storage: Storage, prefix: str, registry: dict[str, type[Model]]) -> Model:
    meta = json.loads(storage.read_text(f"{prefix}/metadata.json"))
    cls = registry[meta["model"]]
    return cls.from_files({name: storage.read_text(f"{prefix}/{name}") for name in meta["files"]})
