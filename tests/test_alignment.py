import numpy as np
import pytest

from rmi import fit_procrustes


def random_orthogonal(dim: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q @ np.diag(np.sign(np.diag(r)))


def test_recovers_exact_square_transform_and_offset() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(500, 8))
    rotation = random_orthogonal(8)
    scale = 1.7
    offset = rng.normal(size=8)
    target = scale * source @ rotation + offset
    fitted = fit_procrustes(source, target, scale="spread")
    np.testing.assert_allclose(fitted.transform(source), target, atol=1e-10)
    assert fitted.orthogonality_error < 1e-10
    assert fitted.scale == pytest.approx(scale)


def test_inverse_round_trip_for_square_map() -> None:
    rng = np.random.default_rng(12)
    source = rng.normal(size=(80, 5))
    target = 0.6 * source @ random_orthogonal(5, seed=13) + 2.0
    fitted = fit_procrustes(source, target)
    np.testing.assert_allclose(fitted.inverse_transform(fitted.transform(source)), source, atol=1e-10)


def test_single_vector_and_batch_have_consistent_shapes() -> None:
    rng = np.random.default_rng(14)
    source = rng.normal(size=(40, 4))
    target = source @ random_orthogonal(4, seed=15)
    fitted = fit_procrustes(source, target)
    assert fitted.transform(source[0]).shape == (4,)
    assert fitted.transform(source).shape == (40, 4)
    np.testing.assert_allclose(fitted.transform(source[0]), fitted.transform(source)[0])


def test_rectangular_map_has_expected_shape_and_constraint() -> None:
    rng = np.random.default_rng(16)
    source = rng.normal(size=(100, 7))
    target = rng.normal(size=(100, 4))
    fitted = fit_procrustes(source, target)
    assert fitted.rotation.shape == (7, 4)
    assert fitted.orthogonality_error < 1e-10
    assert fitted.transform(source).shape == target.shape
    with pytest.raises(ValueError, match="square map"):
        fitted.inverse_transform(target)


def test_wide_target_uses_orthonormal_rows() -> None:
    rng = np.random.default_rng(17)
    source = rng.normal(size=(100, 3))
    target = rng.normal(size=(100, 6))
    fitted = fit_procrustes(source, target)
    assert fitted.rotation.shape == (3, 6)
    assert fitted.orthogonality_error < 1e-10


def test_least_squares_scale_mode() -> None:
    rng = np.random.default_rng(18)
    source = rng.normal(size=(300, 6))
    target = 2.25 * source @ random_orthogonal(6, seed=19)
    fitted = fit_procrustes(source, target, scale="least_squares")
    assert fitted.scale == pytest.approx(2.25, abs=1e-10)
    np.testing.assert_allclose(fitted.transform(source), target, atol=1e-10)


@pytest.mark.parametrize(
    ("source", "target", "message"),
    [
        (np.ones((5, 2)), np.ones((5, 2)), "zero centered variance"),
        (np.ones((5, 2)), np.arange(10).reshape(5, 2), "zero centered variance"),
        (np.zeros((1, 2)), np.zeros((1, 2)), "At least two"),
        (np.zeros((4, 2)), np.zeros((5, 2)), "same sample count"),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), np.ones((2, 2)), "NaN"),
    ],
)
def test_rejects_invalid_inputs(source, target, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_procrustes(source, target)


def test_reports_rank_deficient_calibration() -> None:
    x = np.arange(10, dtype=float).reshape(5, 2)
    source = np.column_stack([x[:, 0], x[:, 0]])
    target = np.column_stack([x[:, 0], x[:, 0]])
    fitted = fit_procrustes(source, target)
    assert fitted.rank < 2


def test_rejects_invalid_scale_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported scale"):
        fit_procrustes(np.eye(3), np.eye(3), scale="unknown")  # type: ignore[arg-type]
