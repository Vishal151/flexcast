"""Shared unit-test fixtures."""

from collections.abc import Callable
from datetime import date

import pytest

from flexcast.history import History
from tests.unit.synthetic import make_history


@pytest.fixture(scope="session")
def synthetic_history() -> Callable[[date, date], History]:
    cache: dict[tuple[date, date], History] = {}

    def get(first: date, last: date) -> History:
        if (first, last) not in cache:
            cache[(first, last)] = make_history(first, last)
        return cache[(first, last)]

    return get
