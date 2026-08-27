from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SyntheticDataset:
    features: np.ndarray
    production_costs: np.ndarray

def generate_contextual_cost_data(
    n_samples: int, *, seed: int, feature_dim: int = 10
) -> SyntheticDataset:
    if n_samples < 1 or feature_dim != 10:
        raise ValueError("benchmark expects n_samples >= 1 and feature_dim = 10")
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, feature_dim))
    base = np.array([52.0, 61.0, 48.0])[:, None]
    base = base + np.array([1.0, -2.0, 3.0, 6.0, 2.0, 0.0])[None, :]
    period_axis = np.linspace(-1.0, 1.0, 6)
    pattern_1 = np.array([1.3, 0.9, 0.4, -0.2, -0.8, -1.1])
    pattern_2 = np.array([-0.8, -0.2, 0.5, 1.1, 0.7, -0.4])
    interaction = np.array([
        [3, -2, 1, 0, -1, 2],
        [-1, 2, -3, 1, 2, 0],
        [2, 0, 1, -2, 3, -1],
    ], dtype=float)
    costs = np.empty((n_samples, 18), dtype=np.float32)
    for i, z in enumerate(features):
        c = base.copy()
        c += 7.0 * z[0] * pattern_1[None, :]
        c += 5.0 * z[1] * pattern_2[None, :]
        c += np.array([5.5, -2.0, 3.0])[:, None] * z[2]
        c += np.array([-1.0, 4.5, 2.0])[:, None] * z[3]
        c += 4.0 * np.tanh(z[4] + 1.2 * period_axis)[None, :]
        c += 3.0 * np.sin(z[5] + np.arange(6) * 0.8)[None, :]
        c += z[6] * z[7] * interaction
        nuisance = 5.0 * (z[8] ** 2 - 1.0)
        c += nuisance * np.array([[1.0] * 6, [0.8] * 6, [1.2] * 6])
        c += 2.0 * z[9]
        c += rng.normal(0.0, 1.2, size=c.shape)
        costs[i] = c.ravel().astype(np.float32)
    return SyntheticDataset(features.astype(np.float32), costs)
