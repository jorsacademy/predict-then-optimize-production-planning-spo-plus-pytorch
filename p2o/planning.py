from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog

@dataclass(frozen=True)
class PlanningSpec:
    demand: np.ndarray
    capacity_hours: np.ndarray
    processing_hours: np.ndarray
    production_upper_bounds: np.ndarray
    inventory_upper_bounds: np.ndarray
    holding_costs: np.ndarray

    def __post_init__(self) -> None:
        demand = np.asarray(self.demand, dtype=float)
        capacity = np.asarray(self.capacity_hours, dtype=float)
        processing = np.asarray(self.processing_hours, dtype=float)
        prod_ub = np.asarray(self.production_upper_bounds, dtype=float)
        inv_ub = np.asarray(self.inventory_upper_bounds, dtype=float)
        holding = np.asarray(self.holding_costs, dtype=float)
        if demand.ndim != 2:
            raise ValueError("demand must have shape [products, periods]")
        products, periods = demand.shape
        if capacity.shape != (periods,) or processing.shape != (products,):
            raise ValueError("capacity or processing shape mismatch")
        for name, array in (
            ("production_upper_bounds", prod_ub),
            ("inventory_upper_bounds", inv_ub),
            ("holding_costs", holding),
        ):
            if array.shape != (products, periods):
                raise ValueError(f"{name} shape mismatch")
        if np.any(demand < 0) or np.any(capacity <= 0) or np.any(processing <= 0):
            raise ValueError("invalid nonpositive planning data")
        if np.any(prod_ub < 0) or np.any(inv_ub < 0) or np.any(holding < 0):
            raise ValueError("bounds and holding costs must be nonnegative")

    @property
    def products(self) -> int:
        return int(self.demand.shape[0])

    @property
    def periods(self) -> int:
        return int(self.demand.shape[1])

    @property
    def production_variables(self) -> int:
        return self.products * self.periods

@dataclass(frozen=True)
class SolveResult:
    production: np.ndarray
    inventory: np.ndarray
    objective: float
    max_violation: float

    @property
    def vector(self) -> np.ndarray:
        return np.concatenate([self.production.ravel(), self.inventory.ravel()])

def default_planning_spec() -> PlanningSpec:
    return PlanningSpec(
        demand=np.array([
            [18, 22, 26, 28, 24, 20],
            [12, 14, 18, 20, 19, 17],
            [10, 12, 13, 16, 18, 15],
        ], dtype=float),
        capacity_hours=np.array([55, 58, 60, 62, 58, 54], dtype=float),
        processing_hours=np.array([1.00, 1.25, 0.80], dtype=float),
        production_upper_bounds=np.array([
            [36] * 6, [30] * 6, [28] * 6,
        ], dtype=float),
        inventory_upper_bounds=np.array([
            [55] * 6, [45] * 6, [40] * 6,
        ], dtype=float),
        holding_costs=np.array([
            [1.50] * 6, [1.20] * 6, [1.00] * 6,
        ], dtype=float),
    )

class ProductionPlanningLP:
    def __init__(self, spec: PlanningSpec):
        self.spec = spec
        p, t = spec.products, spec.periods
        self.n_prod = p * t
        self.n_vars = 2 * self.n_prod

        def x_idx(product: int, period: int) -> int:
            return product * t + period

        def i_idx(product: int, period: int) -> int:
            return self.n_prod + product * t + period

        a_eq, b_eq = [], []
        for product in range(p):
            for period in range(t):
                row = np.zeros(self.n_vars)
                row[x_idx(product, period)] = 1.0
                row[i_idx(product, period)] = -1.0
                if period > 0:
                    row[i_idx(product, period - 1)] = 1.0
                a_eq.append(row)
                b_eq.append(float(spec.demand[product, period]))
            row = np.zeros(self.n_vars)
            row[i_idx(product, t - 1)] = 1.0
            a_eq.append(row)
            b_eq.append(0.0)

        a_ub, b_ub = [], []
        for period in range(t):
            row = np.zeros(self.n_vars)
            for product in range(p):
                row[x_idx(product, period)] = spec.processing_hours[product]
            a_ub.append(row)
            b_ub.append(float(spec.capacity_hours[period]))

        bounds = []
        for product in range(p):
            for period in range(t):
                bounds.append((0.0, float(spec.production_upper_bounds[product, period])))
        for product in range(p):
            for period in range(t):
                bounds.append((0.0, float(spec.inventory_upper_bounds[product, period])))

        self.a_eq = np.asarray(a_eq)
        self.b_eq = np.asarray(b_eq)
        self.a_ub = np.asarray(a_ub)
        self.b_ub = np.asarray(b_ub)
        self.bounds = tuple(bounds)
        self.holding_cost_vector = np.asarray(spec.holding_costs).ravel()

    def full_objective(self, production_costs: np.ndarray) -> np.ndarray:
        production_costs = np.asarray(production_costs, dtype=float)
        if production_costs.shape != (self.n_prod,):
            raise ValueError("production-cost vector has wrong shape")
        return np.concatenate([production_costs, self.holding_cost_vector])

    def max_constraint_violation(self, vector: np.ndarray) -> float:
        vector = np.asarray(vector, dtype=float)
        eq = float(np.max(np.abs(self.a_eq @ vector - self.b_eq)))
        ub = float(np.max(np.maximum(self.a_ub @ vector - self.b_ub, 0.0)))
        lower = float(np.max(np.maximum(-vector, 0.0)))
        upper = 0.0
        for value, (_, hi) in zip(vector, self.bounds):
            upper = max(upper, max(float(value - hi), 0.0))
        return max(eq, ub, lower, upper)

    def solve(self, production_costs: np.ndarray) -> SolveResult:
        result = linprog(
            self.full_objective(production_costs),
            A_ub=self.a_ub, b_ub=self.b_ub,
            A_eq=self.a_eq, b_eq=self.b_eq,
            bounds=self.bounds, method="highs",
        )
        if not result.success:
            raise RuntimeError(f"production-planning LP failed: {result.message}")
        vector = np.asarray(result.x)
        p, t = self.spec.products, self.spec.periods
        production = vector[:self.n_prod].reshape(p, t)
        inventory = vector[self.n_prod:].reshape(p, t)
        violation = self.max_constraint_violation(vector)
        if violation > 1e-7:
            raise RuntimeError(f"post-solve feasibility audit failed: {violation:.3e}")
        return SolveResult(production, inventory, float(result.fun), violation)

    def solve_many(self, production_costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        solutions, objectives = [], []
        for cost in np.asarray(production_costs, dtype=float):
            result = self.solve(cost)
            solutions.append(result.vector)
            objectives.append(result.objective)
        return np.asarray(solutions), np.asarray(objectives)
