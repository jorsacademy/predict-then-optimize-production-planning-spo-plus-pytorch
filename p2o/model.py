from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .planning import ProductionPlanningLP

class CostPredictor(nn.Module):
    def __init__(self, feature_dim: int, cost_center: np.ndarray, cost_scale: np.ndarray):
        super().__init__()
        output_dim = len(cost_center)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 20), nn.Tanh(),
            nn.Linear(20, 5), nn.Tanh(),
            nn.Linear(5, output_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.register_buffer("cost_center", torch.as_tensor(cost_center, dtype=torch.float32))
        self.register_buffer("cost_scale", torch.as_tensor(cost_scale, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.cost_center + self.cost_scale * self.network(features)

class SPOPlusLoss(nn.Module):
    def __init__(self, planner: ProductionPlanningLP):
        super().__init__()
        self.planner = planner

    def forward(
        self,
        predicted_costs: torch.Tensor,
        true_costs: torch.Tensor,
        true_solutions: torch.Tensor,
    ) -> torch.Tensor:
        if predicted_costs.shape != true_costs.shape:
            raise ValueError("predicted and true costs must have identical shape")
        q = (2.0 * predicted_costs.detach() - true_costs.detach()).cpu().numpy()
        perturbed, _ = self.planner.solve_many(q)
        wq = torch.as_tensor(perturbed, dtype=predicted_costs.dtype, device=predicted_costs.device)
        n = self.planner.n_prod
        holding = torch.as_tensor(
            self.planner.holding_cost_vector,
            dtype=predicted_costs.dtype,
            device=predicted_costs.device,
        )
        transformed = 2.0 * predicted_costs - true_costs
        per_instance = (
            (transformed * (true_solutions[:, :n] - wq[:, :n])).sum(1)
            + (holding * (true_solutions[:, n:] - wq[:, n:])).sum(1)
        )
        return per_instance.mean()

def decision_activity_weights(true_solutions: np.ndarray, n_prod: int) -> np.ndarray:
    activity = np.std(np.asarray(true_solutions)[:, :n_prod], axis=0)
    mean_activity = float(activity.mean())
    if mean_activity <= 1e-12:
        return np.ones(n_prod, dtype=np.float32)
    return (0.25 + 0.75 * activity / mean_activity).astype(np.float32)
