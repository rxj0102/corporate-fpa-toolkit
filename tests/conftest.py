import numpy as np
import pytest

from corporate_fpa_toolkit.driver_based_budget import default_business_units


@pytest.fixture
def budget_units():
    return default_business_units()


@pytest.fixture(autouse=True)
def _reset_random_seed():
    np.random.seed(0)
