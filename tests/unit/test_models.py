from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flexcast.features import FEATURE_COLUMNS
from flexcast.models.base import COLUMNS, Model, load_model, pinball_loss, save_model
from flexcast.models.baselines import NaiveLastDay, SeasonalNaive
from flexcast.storage import Storage


def feature_frame(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(50, 20, (n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    X["hour"] = np.tile(np.arange(48) / 2, n // 48 + 1)[:n]
    y = X["price_d2"] * 0.6 + 30 * np.sin(X["hour"] / 24 * 2 * np.pi) + rng.normal(0, 10, n)
    return X, pd.Series(y, name="price")


def test_pinball_loss_hand_value() -> None:
    y = pd.Series([10.0])
    pred = pd.DataFrame({"p10": [12.0], "p50": [10.0], "p90": [8.0]})
    # p10 over by 2: (0.1-1)*-2 = 1.8; p50 exact: 0; p90 under by 2: 0.9*2 = 1.8 -> mean 1.2
    assert pinball_loss(y, pred) == pytest.approx(1.2)


@pytest.mark.parametrize(
    ("model", "source", "fallback"),
    [
        (NaiveLastDay(), "price_d2", "price_d2_mean"),
        (SeasonalNaive(), "price_d7", "price_roll7_mean"),
    ],
)
def test_baselines_copy_lagged_price_with_fallback(
    model: Model, source: str, fallback: str
) -> None:
    X, _ = feature_frame(48)
    X.loc[X.index[3], source] = np.nan
    pred = model.predict(X)
    assert list(pred.columns) == COLUMNS and pred.notna().all().all()
    assert pred["p50"].iloc[0] == X[source].iloc[0]
    assert pred["p50"].iloc[3] == X[fallback].iloc[3]
    assert (pred["p10"] == pred["p90"]).all()


def round_trip(model: Model, tmp_path: Path) -> Model:
    from flexcast.models.registry import MODELS

    storage = Storage(f"file://{tmp_path}")
    save_model(storage, model, "models/x", {"train_end": "2025-01-01"})
    return load_model(storage, "models/x", MODELS)


@pytest.mark.parametrize("model", [NaiveLastDay(), SeasonalNaive()])
def test_baselines_round_trip(model: Model, tmp_path: Path) -> None:
    X, _ = feature_frame(48)
    loaded = round_trip(model, tmp_path)
    assert type(loaded) is type(model)
    pd.testing.assert_frame_equal(loaded.predict(X), model.predict(X))


# --- LightGBM --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_lgbm() -> Model:
    from flexcast.models.lgbm import LGBMQuantile

    X, y = feature_frame()
    m = LGBMQuantile()
    m.fit(X, y)
    return m


def test_lgbm_quantiles_ordered_and_complete(fitted_lgbm: Model) -> None:
    X, _ = feature_frame(seed=1)
    pred = fitted_lgbm.predict(X)
    assert pred.notna().all().all()
    assert (pred["p10"] <= pred["p50"]).all() and (pred["p50"] <= pred["p90"]).all()
    assert (pred["p90"] - pred["p10"]).mean() > 1  # a real interval, not collapsed


def test_lgbm_beats_naive_on_learnable_signal(fitted_lgbm: Model) -> None:
    X, y = feature_frame(seed=1)
    lgbm_mae = (fitted_lgbm.predict(X)["p50"] - y).abs().mean()
    naive_mae = (NaiveLastDay().predict(X)["p50"] - y).abs().mean()
    assert lgbm_mae < naive_mae


def test_lgbm_round_trip_predicts_identically(fitted_lgbm: Model, tmp_path: Path) -> None:
    X, _ = feature_frame(seed=2)
    loaded = round_trip(fitted_lgbm, tmp_path)
    pd.testing.assert_frame_equal(loaded.predict(X), fitted_lgbm.predict(X))
    files = sorted(p.name for p in (tmp_path / "models/x").iterdir())
    assert files == ["metadata.json", "p10.txt", "p50.txt", "p90.txt", "params.json"]


def test_lgbm_training_is_deterministic() -> None:
    from flexcast.models.lgbm import LGBMQuantile

    X, y = feature_frame()
    a, b = LGBMQuantile(), LGBMQuantile()
    a.fit(X, y)
    b.fit(X, y)
    pd.testing.assert_frame_equal(a.predict(X), b.predict(X))
