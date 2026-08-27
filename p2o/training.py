from __future__ import annotations
import copy
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t as student_t
from .data import generate_contextual_cost_data
from .model import CostPredictor, SPOPlusLoss, decision_activity_weights
from .planning import ProductionPlanningLP, default_planning_spec

@dataclass(frozen=True)
class PolicyMetrics:
    name: str
    cost_rmse: float
    mean_regret: float
    median_regret: float
    p90_regret: float
    mean_relative_regret_pct: float
    max_feasibility_violation: float

@dataclass(frozen=True)
class ExperimentResult:
    mse: PolicyMetrics
    weighted_mse: PolicyMetrics
    spo_plus: PolicyMetrics
    paired_spo_minus_mse_mean: float
    paired_spo_minus_mse_ci95_low: float
    paired_spo_minus_mse_ci95_high: float
    paired_spo_minus_weighted_mean: float
    paired_spo_minus_weighted_ci95_low: float
    paired_spo_minus_weighted_ci95_high: float

def _batches(n: int, batch_size: int, rng: np.random.Generator):
    indices = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield indices[start:start + batch_size]

def _paired_ci95(d: np.ndarray) -> tuple[float, float, float]:
    d = np.asarray(d, dtype=float)
    mean = float(d.mean())
    half = float(student_t.ppf(0.975, df=len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))
    return mean, mean - half, mean + half

def evaluate_model(name, model, features, true_costs, true_objectives, planner):
    model.eval()
    with torch.no_grad():
        predicted = model(torch.as_tensor(features, dtype=torch.float32)).cpu().numpy()
    rmse = float(np.sqrt(np.mean((predicted - true_costs) ** 2)))
    regrets, relative = [], []
    max_violation = 0.0
    for pred, true, optimum in zip(predicted, true_costs, true_objectives):
        decision = planner.solve(pred)
        realized = float(planner.full_objective(true) @ decision.vector)
        regret = max(realized - float(optimum), 0.0)
        regrets.append(regret)
        relative.append(100.0 * regret / max(abs(float(optimum)), 1e-9))
        max_violation = max(max_violation, decision.max_violation)
    regrets = np.asarray(regrets)
    metrics = PolicyMetrics(
        name, rmse, float(regrets.mean()), float(np.median(regrets)),
        float(np.quantile(regrets, 0.90)), float(np.mean(relative)), max_violation,
    )
    return metrics, regrets

def _mse_epoch(model, optimizer, x, c, *, batch_size, rng, weights=None):
    model.train()
    for idx in _batches(len(x), batch_size, rng):
        pred = model(torch.as_tensor(x[idx], dtype=torch.float32))
        true = torch.as_tensor(c[idx], dtype=torch.float32)
        sq = (pred - true) ** 2
        loss = sq.mean() if weights is None else (sq * weights).mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

def _spo_epoch(model, optimizer, x, c, w, loss_module, *, batch_size, rng):
    model.train()
    for idx in _batches(len(x), batch_size, rng):
        pred = model(torch.as_tensor(x[idx], dtype=torch.float32))
        true = torch.as_tensor(c[idx], dtype=torch.float32)
        sol = torch.as_tensor(w[idx], dtype=torch.float32)
        loss = loss_module(pred, true, sol)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

def train_and_compare(
    *, seed=42, train_samples=480, validation_samples=128, test_samples=160,
    pretrain_epochs=8, finetune_epochs=12, batch_size=32, checkpoint_dir=None,
) -> ExperimentResult:
    if min(train_samples, validation_samples, test_samples) < 2:
        raise ValueError("each split needs at least two samples")
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    planner = ProductionPlanningLP(default_planning_spec())
    data = generate_contextual_cost_data(train_samples + validation_samples + test_samples, seed=seed)
    i1, i2 = train_samples, train_samples + validation_samples
    x_train, x_val, x_test = data.features[:i1], data.features[i1:i2], data.features[i2:]
    c_train, c_val, c_test = data.production_costs[:i1], data.production_costs[i1:i2], data.production_costs[i2:]
    w_train, _ = planner.solve_many(c_train)
    _, z_val = planner.solve_many(c_val)
    _, z_test = planner.solve_many(c_test)

    center, scale = c_train.mean(0), c_train.std(0) + 1e-3
    common = CostPredictor(x_train.shape[1], center, scale)
    common_opt = torch.optim.Adam(common.parameters(), lr=2e-3)
    for _ in range(pretrain_epochs):
        _mse_epoch(common, common_opt, x_train, c_train, batch_size=max(batch_size, 32), rng=rng)

    mse, weighted, spo = copy.deepcopy(common), copy.deepcopy(common), copy.deepcopy(common)
    mse_opt = torch.optim.Adam(mse.parameters(), lr=1e-3)
    weighted_opt = torch.optim.Adam(weighted.parameters(), lr=1e-3)
    spo_opt = torch.optim.Adam(spo.parameters(), lr=4e-4)
    weights = torch.as_tensor(decision_activity_weights(w_train, planner.n_prod))
    spo_loss = SPOPlusLoss(planner)
    best_mse, best_weighted, best_spo = copy.deepcopy(mse.state_dict()), copy.deepcopy(weighted.state_dict()), copy.deepcopy(spo.state_dict())
    best_mse_rmse = best_weighted_loss = best_spo_regret = float("inf")

    for epoch in range(1, finetune_epochs + 1):
        _mse_epoch(mse, mse_opt, x_train, c_train, batch_size=batch_size, rng=rng)
        _mse_epoch(weighted, weighted_opt, x_train, c_train, batch_size=batch_size, rng=rng, weights=weights)
        _spo_epoch(spo, spo_opt, x_train, c_train, w_train, spo_loss, batch_size=batch_size, rng=rng)
        mse_val, _ = evaluate_model("mse", mse, x_val, c_val, z_val, planner)
        weighted_val, _ = evaluate_model("decision_activity_weighted_mse", weighted, x_val, c_val, z_val, planner)
        spo_val, _ = evaluate_model("spo_plus", spo, x_val, c_val, z_val, planner)
        if mse_val.cost_rmse < best_mse_rmse:
            best_mse_rmse, best_mse = mse_val.cost_rmse, copy.deepcopy(mse.state_dict())
        weighted.eval()
        with torch.no_grad():
            pred = weighted(torch.as_tensor(x_val, dtype=torch.float32))
            target = torch.as_tensor(c_val, dtype=torch.float32)
            weighted_loss = float((((pred - target) ** 2) * weights).mean())
        if weighted_loss < best_weighted_loss:
            best_weighted_loss, best_weighted = weighted_loss, copy.deepcopy(weighted.state_dict())
        if spo_val.mean_regret < best_spo_regret:
            best_spo_regret, best_spo = spo_val.mean_regret, copy.deepcopy(spo.state_dict())
        print(
            f"epoch={epoch:02d} mse_rmse={mse_val.cost_rmse:.3f} "
            f"mse_regret={mse_val.mean_regret:.3f} weighted_regret={weighted_val.mean_regret:.3f} "
            f"spo_regret={spo_val.mean_regret:.3f}"
        )

    mse.load_state_dict(best_mse); weighted.load_state_dict(best_weighted); spo.load_state_dict(best_spo)
    if checkpoint_dir:
        out = Path(checkpoint_dir); out.mkdir(parents=True, exist_ok=True)
        torch.save(mse.state_dict(), out / "mse_predictor.pt")
        torch.save(weighted.state_dict(), out / "weighted_mse_predictor.pt")
        torch.save(spo.state_dict(), out / "spo_plus_predictor.pt")

    mse_m, mse_r = evaluate_model("mse", mse, x_test, c_test, z_test, planner)
    weighted_m, weighted_r = evaluate_model("decision_activity_weighted_mse", weighted, x_test, c_test, z_test, planner)
    spo_m, spo_r = evaluate_model("spo_plus", spo, x_test, c_test, z_test, planner)
    dm = _paired_ci95(spo_r - mse_r)
    dw = _paired_ci95(spo_r - weighted_r)
    return ExperimentResult(mse_m, weighted_m, spo_m, *dm, *dw)
