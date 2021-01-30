"""Tests for unfoldNd/__init__.py."""

import unfoldNd
import pytest
import time

NAMES = ["world", "github"]
IDS = NAMES


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_hello(name):
    """Test hello function."""
    unfoldNd.hello(name)


@pytest.mark.expensive
@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_hello_expensive(name):
    """Expensive test of hello function. Will only be run on main and development."""
    time.sleep(1)
    unfoldNd.hello(name)
