from __future__ import annotations
import argparse
import math
import numpy as np
import torch
from p2o import (
    PlanningSpec, ProductionPlanningLP, SPOPlusLoss, default_planning_spec,
    generate_contextual_cost_data, train_and_compare,
)

def self_test() -> None:
    spec = PlanningSpec(
        demand=np.array([[0.0, 10.0]]), capacity_hours=np.array([10.0, 10.0]),
        processing_hours=np.array([1.0]), production_upper_bounds=np.array([[10.0, 10.0]]),
        inventory_upper_bounds=np.array([[10.0, 10.0]]), holding_costs=np.array([[1.0, 1.0]]),
    )
    planner = ProductionPlanningLP(spec)
    result = planner.solve(np.array([2.0, 10.0]))
    assert np.allclose(result.production, [[10.0, 0.0]], atol=1e-9)
    assert np.allclose(result.inventory, [[10.0, 0.0]], atol=1e-9)
    assert math.isclose(result.objective, 30.0, abs_tol=1e-9)

    true = torch.tensor([[2.0, 10.0]])
    pred = torch.tensor([[10.0, 2.0]], requires_grad=True)
    w_true = torch.tensor(result.vector[None, :], dtype=torch.float32)
    loss = SPOPlusLoss(planner)(pred, true, w_true)
    loss.backward()
    q_solution = planner.solve((2.0 * pred.detach() - true).numpy()[0])
    expected = 2.0 * (result.production.ravel() - q_solution.production.ravel())
    assert np.allclose(pred.grad.numpy()[0], expected, atol=1e-6)
    assert float(loss.detach()) >= -1e-7

    a = generate_contextual_cost_data(5, seed=123)
    b = generate_contextual_cost_data(5, seed=123)
    assert np.array_equal(a.features, b.features)
    assert np.array_equal(a.production_costs, b.production_costs)
    default_solution = ProductionPlanningLP(default_planning_spec()).solve(a.production_costs[0])
    assert default_solution.max_violation <= 1e-7
    print("Predict-then-optimize production planning SPO+ self-test: OK")

def _print_metrics(m) -> None:
    print(
        f"{m.name:<33} RMSE={m.cost_rmse:7.3f} mean_regret={m.mean_regret:9.3f} "
        f"median={m.median_regret:8.3f} p90={m.p90_regret:8.3f} "
        f"relative={m.mean_relative_regret_pct:6.3f}%"
    )

def print_experiment(r) -> None:
    print("=" * 112)
    print("PREDICT-THEN-OPTIMIZE PRODUCTION PLANNING — MSE vs WEIGHTED MSE vs SPO+")
    print("=" * 112)
    for m in (r.mse, r.weighted_mse, r.spo_plus):
        _print_metrics(m)
    print()
    print(
        f"paired regret difference SPO+ - MSE       : {r.paired_spo_minus_mse_mean:.3f} "
        f"[95% CI {r.paired_spo_minus_mse_ci95_low:.3f}, {r.paired_spo_minus_mse_ci95_high:.3f}]"
    )
    print(
        f"paired regret difference SPO+ - weighted  : {r.paired_spo_minus_weighted_mean:.3f} "
        f"[95% CI {r.paired_spo_minus_weighted_ci95_low:.3f}, {r.paired_spo_minus_weighted_ci95_high:.3f}]"
    )
    print("Negative paired differences favor SPO+; each downstream LP is solved exactly to solver tolerance.")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-samples", type=int, default=480)
    p.add_argument("--validation-samples", type=int, default=128)
    p.add_argument("--test-samples", type=int, default=160)
    p.add_argument("--pretrain-epochs", type=int, default=8)
    p.add_argument("--finetune-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        print_experiment(train_and_compare(
            seed=args.seed, train_samples=args.train_samples,
            validation_samples=args.validation_samples, test_samples=args.test_samples,
            pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs,
            batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir,
        ))
