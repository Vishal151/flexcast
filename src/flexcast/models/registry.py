"""Name -> class, for loading saved models. LightGBM is imported lazily on first use."""

from flexcast.models.base import Model
from flexcast.models.baselines import (
    NaiveDayAhead,
    NaiveLastDay,
    SeasonalNaive,
    SeasonalNaiveDayAhead,
)


class _Registry(dict[str, type[Model]]):
    def __missing__(self, name: str) -> type[Model]:
        if name == "lgbm_quantile":
            from flexcast.models.lgbm import LGBMQuantile

            self[name] = LGBMQuantile
            return LGBMQuantile
        raise KeyError(name)


MODELS: dict[str, type[Model]] = _Registry(
    {
        NaiveLastDay.name: NaiveLastDay,
        SeasonalNaive.name: SeasonalNaive,
        NaiveDayAhead.name: NaiveDayAhead,
        SeasonalNaiveDayAhead.name: SeasonalNaiveDayAhead,
    }
)
