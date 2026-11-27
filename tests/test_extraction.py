import numpy as np
import pytest

from rmi.extraction import pool_token_activations


def test_mean_pooling_ignores_padding():
    hidden = np.array([[[1, 2], [3, 4], [99, 99]]], dtype=float)
    mask = np.array([[1, 1, 0]])
    np.testing.assert_allclose(pool_token_activations(hidden, mask), [[2, 3]])


def test_last_token_pooling_supports_left_padding():
    hidden = np.array([[[0, 0], [2, 3], [4, 5]]], dtype=float)
    mask = np.array([[0, 1, 1]])
    np.testing.assert_allclose(
        pool_token_activations(hidden, mask, pooling="last_token"), [[4, 5]]
    )


def test_cls_pooling_uses_first_token():
    hidden = np.array([[[7, 8], [2, 3]]], dtype=float)
    mask = np.array([[1, 1]])
    np.testing.assert_allclose(pool_token_activations(hidden, mask, pooling="cls"), [[7, 8]])


@pytest.mark.parametrize(
    ("hidden", "mask", "pooling"),
    [
        (np.zeros((2, 3)), np.ones((2, 3)), "mean"),
        (np.zeros((1, 3, 2)), np.ones((1, 2)), "mean"),
        (np.zeros((1, 3, 2)), np.zeros((1, 3)), "mean"),
        (np.zeros((1, 3, 2)), np.ones((1, 3)), "unknown"),
    ],
)
def test_invalid_pooling_inputs(hidden, mask, pooling):
    with pytest.raises(ValueError):
        pool_token_activations(hidden, mask, pooling=pooling)
