import math
import unittest

import numpy as np
import torch

from p2o import (
    CostPredictor,
    PlanningSpec,
    ProductionPlanningLP,
    SPOPlusLoss,
    decision_activity_weights,
    default_planning_spec,
    generate_contextual_cost_data,
    train_and_compare,
)


class PredictThenOptimizeTests(unittest.TestCase):
    def test_hand_solvable_production_shift(self):
        spec = PlanningSpec(
            demand=np.array([[0.0, 10.0]]),
            capacity_hours=np.array([10.0, 10.0]),
            processing_hours=np.array([1.0]),
            production_upper_bounds=np.array([[10.0, 10.0]]),
            inventory_upper_bounds=np.array([[10.0, 10.0]]),
            holding_costs=np.array([[1.0, 1.0]]),
        )
        planner = ProductionPlanningLP(spec)
        result = planner.solve(np.array([2.0, 10.0]))
        np.testing.assert_allclose(result.production, [[10.0, 0.0]], atol=1e-9)
        np.testing.assert_allclose(result.inventory, [[10.0, 0.0]], atol=1e-9)
        self.assertTrue(math.isclose(result.objective, 30.0, abs_tol=1e-9))

    def test_postsolve_feasibility_audit(self):
        planner = ProductionPlanningLP(default_planning_spec())
        costs = generate_contextual_cost_data(4, seed=4).production_costs
        for cost in costs:
            self.assertLessEqual(planner.solve(cost).max_violation, 1e-7)

    def test_data_generation_is_reproducible(self):
        a = generate_contextual_cost_data(10, seed=8)
        b = generate_contextual_cost_data(10, seed=8)
        np.testing.assert_array_equal(a.features, b.features)
        np.testing.assert_array_equal(a.production_costs, b.production_costs)

    def test_spo_plus_gradient_matches_analytical_solution_difference(self):
        spec = PlanningSpec(
            demand=np.array([[0.0, 10.0]]),
            capacity_hours=np.array([10.0, 10.0]),
            processing_hours=np.array([1.0]),
            production_upper_bounds=np.array([[10.0, 10.0]]),
            inventory_upper_bounds=np.array([[10.0, 10.0]]),
            holding_costs=np.array([[1.0, 1.0]]),
        )
        planner = ProductionPlanningLP(spec)
        true = torch.tensor([[2.0, 10.0]])
        pred = torch.tensor([[10.0, 2.0]], requires_grad=True)
        true_solution = planner.solve(true.numpy()[0])
        w_true = torch.tensor(true_solution.vector[None, :], dtype=torch.float32)
        loss = SPOPlusLoss(planner)(pred, true, w_true)
        loss.backward()

        q = (2.0 * pred.detach() - true).numpy()[0]
        q_solution = planner.solve(q)
        expected = 2.0 * (
            true_solution.production.ravel() - q_solution.production.ravel()
        )
        np.testing.assert_allclose(pred.grad.numpy()[0], expected, atol=1e-6)
        self.assertGreaterEqual(float(loss.detach()), -1e-7)

    def test_cost_predictor_shape_and_gradients(self):
        data = generate_contextual_cost_data(8, seed=9)
        center = data.production_costs.mean(axis=0)
        scale = data.production_costs.std(axis=0) + 1e-3
        model = CostPredictor(10, center, scale)
        x = torch.tensor(data.features)
        y = model(x)
        self.assertEqual(tuple(y.shape), (8, 18))
        y.mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_decision_activity_weights_are_positive(self):
        planner = ProductionPlanningLP(default_planning_spec())
        costs = generate_contextual_cost_data(12, seed=11).production_costs
        solutions, _ = planner.solve_many(costs)
        weights = decision_activity_weights(solutions, planner.n_prod)
        self.assertEqual(weights.shape, (18,))
        self.assertTrue(np.all(weights > 0))

    def test_true_optimum_has_zero_self_regret(self):
        planner = ProductionPlanningLP(default_planning_spec())
        cost = generate_contextual_cost_data(1, seed=15).production_costs[0]
        result = planner.solve(cost)
        realized = planner.full_objective(cost) @ result.vector
        self.assertTrue(math.isclose(realized, result.objective, abs_tol=1e-8))

    def test_short_end_to_end_training_smoke(self):
        result = train_and_compare(
            seed=21,
            train_samples=48,
            validation_samples=16,
            test_samples=20,
            pretrain_epochs=1,
            finetune_epochs=1,
            batch_size=16,
        )
        for metrics in (result.mse, result.weighted_mse, result.spo_plus):
            self.assertTrue(math.isfinite(metrics.cost_rmse))
            self.assertTrue(math.isfinite(metrics.mean_regret))
            self.assertGreaterEqual(metrics.mean_regret, -1e-8)
            self.assertLessEqual(metrics.max_feasibility_violation, 1e-7)


if __name__ == "__main__":
    unittest.main()
