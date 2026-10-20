import pytest

from src.train import learning_rate


def test_learning_rate_warms_up_and_finishes_at_one_percent():
    assert learning_rate(0, 180, 5, 0.0015) == pytest.approx(0.0003)
    assert learning_rate(4, 180, 5, 0.0015) == pytest.approx(0.0015)
    assert learning_rate(5, 180, 5, 0.0015) == pytest.approx(0.0015)
    assert learning_rate(179, 180, 5, 0.0015) == pytest.approx(0.000015)
