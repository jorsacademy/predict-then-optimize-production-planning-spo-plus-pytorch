# Predict-then-Optimize Production Planning with SPO+ in PyTorch

A from-scratch decision-focused learning project for contextual multi-period production planning. A neural predictor estimates uncertain production-cost coefficients, an exact LP converts those coefficients into production/inventory decisions, and SPO+ trains the predictor against downstream decision quality rather than coefficient error alone.

This repository does **not** copy PyEPO source code, class structure, solver abstractions, datasets, or notebooks. PyEPO is listed under Related Work because it is an important general-purpose predict-then-optimize library. The SPO+ loss in this project is independently implemented from the mathematical formulation in Elmachtoub & Grigas.

## Decision pipeline

```text
context features
      ↓
PyTorch cost predictor
      ↓
predicted production-cost vector
      ↓
exact multi-period production-planning LP
      ↓
production + inventory decision
      ↓
realized cost under the true coefficients
      ↓
decision regret
```

The core question is deliberately different from ordinary regression:

> Does a model with better coefficient RMSE necessarily produce better operational decisions?

## Production-planning LP

The benchmark contains three products and six periods. Decisions are:

```text
x[p,t]  production quantity
I[p,t]  end-of-period inventory
```

Inventory balance:

```text
I[p,t-1] + x[p,t] - I[p,t] = demand[p,t]
```

Capacity:

```text
sum_p processing_hours[p] * x[p,t] <= capacity[t]
```

The model also includes product/period production upper bounds, inventory bounds, nonnegative variables, known holding costs, and zero final inventory.

The feasible region is fixed across the dataset. Only the 18 production-cost coefficients vary with context, which is the standard fixed-feasible-region setting for SPO+.

All downstream solves use `scipy.optimize.linprog(method="highs")`. A separate post-solve audit checks balance equations, capacity, and variable bounds.

## Synthetic contextual cost model

Each observation contains ten exogenous features. True production costs combine:

- period-specific energy-like shocks;
- product-specific material/labor effects;
- nonlinear `tanh` and sinusoidal terms;
- feature interactions;
- common-mode product shocks;
- irreducible observation/process noise.

The predictor intentionally has a narrow latent bottleneck. It cannot perfectly reconstruct every coefficient, so the learning objective matters.

## Three training objectives

All three models use the **same predictor architecture** and begin from the **same MSE warm-start**.

### 1. Standard MSE

```text
mean((c_hat - c)^2)
```

The validation checkpoint is selected by coefficient RMSE.

### 2. Decision-activity-weighted MSE

A fixed per-coefficient weight is computed once from the variability of the true optimal production quantities in the training set. This remains a regression loss: it does not differentiate through the optimizer and does not solve a perturbed optimization problem on each gradient step.

### 3. SPO+

For minimization, define:

```text
q = 2 * c_hat - c
```

Let `w(c)` be the true optimal production-planning solution and `w(q)` the exact solution under the transformed production costs. The SPO+ subgradient with respect to the predicted production costs is:

```text
2 * (x(c) - x(q))
```

The implementation obtains `w(q)` by resolving the production-planning LP, treats that discrete/LP optimizer output as fixed during backpropagation, and constructs the analytical surrogate directly in PyTorch. It does not import an SPO+ implementation from PyEPO.

## Decision regret

For true full objective `c` and the decision obtained from predicted costs `c_hat`:

```text
regret(c_hat, c)
=
true_cost(decision(c_hat)) - true_optimal_cost(c)
```

Lower is better. Regret is evaluated with fresh test contexts and exact downstream LP solves.

Reported metrics are:

- coefficient RMSE;
- mean / median / 90th-percentile regret;
- mean relative regret;
- paired regret differences with 95% Student-t confidence intervals;
- maximum post-solve feasibility violation.

A negative paired `SPO+ - baseline` regret difference favors SPO+.

## Development benchmark

A deterministic development run with seed 42, 480 training samples, 128 validation samples, 160 test samples, 8 common warm-start epochs and 12 fine-tuning epochs produced:

```text
standard MSE
  cost RMSE                 10.731
  mean regret              110.565
  median regret             60.694
  p90 regret               250.029
  mean relative regret       0.638%

decision-activity weighted MSE
  cost RMSE                 10.743
  mean regret              110.748
  median regret             63.650
  p90 regret               240.563
  mean relative regret       0.639%

SPO+
  cost RMSE                 10.994
  mean regret              102.739
  median regret             61.792
  p90 regret               240.139
  mean relative regret       0.595%
```

SPO+ therefore had **worse coefficient RMSE but lower mean downstream regret** on this fixed test set. The paired regret differences were:

```text
SPO+ - MSE       = -7.826   95% CI [-16.858, 1.206]
SPO+ - weighted  = -8.009   95% CI [-16.632, 0.615]
```

Negative differences favor SPO+. Both confidence intervals include zero, so this single development run is evidence of a decision-quality trade-off, **not** a statistically conclusive claim that SPO+ universally dominates the regression baselines. All values are consequences of the declared synthetic benchmark, not production savings claims.

## Tests

The regression suite checks:

- a hand-solvable two-period production-shift oracle;
- downstream LP feasibility auditing;
- deterministic data generation;
- analytical SPO+ gradient against `2 * (x_true - x_perturbed)`;
- predictor shapes and gradient flow;
- positivity of decision-activity weights;
- zero regret when the true optimum is evaluated under its own costs;
- an end-to-end MSE / weighted-MSE / SPO+ training smoke run.

## Run

Install:

```bash
pip install -r requirements.txt
```

Self-test:

```bash
python production_planning_spo.py --self-test
```

Regression tests:

```bash
python -m unittest discover -s tests -v
```

Full default experiment:

```bash
python production_planning_spo.py
```

Optionally save the three trained checkpoints:

```bash
python production_planning_spo.py --checkpoint-dir checkpoints
```

## Exactness and claims

HiGHS solves each declared continuous production-planning LP to solver optimality subject to numerical tolerances. This gives an exact downstream optimization oracle for the benchmark LP.

That does **not** mean:

- the neural predictor is globally optimal;
- SPO+ must outperform MSE on every dataset or random seed;
- the synthetic cost model represents a real factory;
- the reported regret difference is a real-world savings estimate.

The point of the repository is to separate predictive accuracy from decision quality under a transparent exact downstream optimizer.

## Related work

- Elmachtoub, A. N. & Grigas, P. **Smart “Predict, then Optimize”**. *Management Science* 68(1), 2022. DOI: `10.1287/mnsc.2020.3922`.
- Tang, B. & Khalil, E. B. **PyEPO: a PyTorch-based end-to-end predict-then-optimize library for linear and integer programming**. *Mathematical Programming Computation*, 2024.
- PyEPO repository: https://github.com/khalil-research/PyEPO

PyEPO is a broad library covering many decision-focused losses and solver backends. This repository instead focuses on one production-planning case study with a deliberately small, transparent, independently implemented training stack.
