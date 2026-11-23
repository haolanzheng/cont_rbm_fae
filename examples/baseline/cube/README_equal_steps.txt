Equal-step comparison on the existing linear cube baseline

GD, SLSQP and Coleman-Li reflective trust region all receive an outer-iteration
budget of 80 per greedy round. All other solver settings match compare_optimizers.
Early convergence or other solver stops are retained and explicitly recorded;
no artificial idle work is added. The final best feasible evaluated point is used.
SQP line-search evaluations and reflective trust-region subproblem solves are
included in time but not counted as additional outer iterations. Equal iteration
budgets are not assertions of equal arithmetic work.

d=4,8,16,32,64,128; seeds 1000,...,1007; N=d/2 for final errors and times.
Error curves also extend through d-1 for d=16,32,64. All methods use the same
existing vertex test set, same model and one CPU thread. Initialization is either
one uniform start or the best of a fixed pool of 256 uniform points, reused over
rounds. Pool generation and scoring are included in training time. Shared HF
setup, tests and disk I/O are excluded from training time.

From the repository root, using the previously installed Fides dependency:
PYTHONPATH=runs/cube_optimizer_comparison_20261115/python_deps:. .venv_cpu/bin/python -u -m examples.baseline.cube.compare_equal_steps
PYTHONPATH=runs/cube_optimizer_comparison_20261115/python_deps:. .venv_cpu/bin/python -u -m examples.baseline.cube.report_equal_steps

Results: runs/cube_optimizer_equal_steps_20261115
summary.csv: error, time and evaluation counts.
iteration_counts.csv: actual iterations and budget/early-stop counts.
results.json: per-run and per-round records; failed/capped runs are retained.
frozen.json: same-H/same-start paired searches with the same 80-iteration cap.
The earlier 500-iteration experiment and all slides remain unchanged.
