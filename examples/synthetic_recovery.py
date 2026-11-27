"""Run a deterministic synthetic Procrustes recovery experiment."""

import numpy as np

from rmi import fit_procrustes


def main() -> None:
    rng = np.random.default_rng(2026)
    n_samples, dimension = 1_000, 16
    source = rng.normal(size=(n_samples, dimension))
    raw = rng.normal(size=(dimension, dimension))
    rotation, _ = np.linalg.qr(raw)
    expected_scale = 1.8
    offset = rng.normal(size=dimension)
    target = expected_scale * source @ rotation + offset

    # Fit on 70%; evaluate once on the held-out 30%.
    split = int(n_samples * 0.7)
    fitted = fit_procrustes(source[:split], target[:split], scale="spread")
    prediction = fitted.transform(source[split:])
    truth = target[split:]
    relative_error = np.linalg.norm(prediction - truth) / np.linalg.norm(truth)

    print(f"n_train={split}")
    print(f"n_test={n_samples - split}")
    print(f"source_dim={fitted.source_dim}")
    print(f"target_dim={fitted.target_dim}")
    print(f"estimated_scale={fitted.scale:.8f}")
    print(f"expected_scale={expected_scale:.8f}")
    print(f"calibration_rank={fitted.rank}")
    print(f"orthogonality_error={fitted.orthogonality_error:.3e}")
    print(f"held_out_relative_error={relative_error:.3e}")


if __name__ == "__main__":
    main()
