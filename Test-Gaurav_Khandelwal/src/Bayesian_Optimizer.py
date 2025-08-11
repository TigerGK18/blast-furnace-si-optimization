"""
bayes_opt_with_pulp.py

Bayesian optimization wrapper that uses a trained model (joblib) as a surrogate
for predicted SI and finds optimal controllable furnace parameters. After
finding the best suggestion from Bayesian Optimization, it uses PuLP to project
that suggestion onto a feasible set defined by linear constraints by minimizing
sum of absolute deviations (L1 distance) — formulated as a linear program.

Usage:
    python bayes_opt_with_pulp.py --model si_xgb_model.pkl --input sample_input.csv \
        --n_iter 25

Dependencies:
    pip install pandas numpy joblib bayesian-optimization pulp scikit-learn

Note: Adjust the `CONTROL_VARS` and `BOUNDS` to match variables you can change
in your process. The input file should contain the same feature columns used for
training the model. The script uses the last row of the input as the baseline
(state for non-optimized features.

"""

import argparse
import json
import numpy as np
import pandas as pd
import joblib
from bayes_opt import BayesianOptimization
import pulp
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# CONTROL VARIABLES & BOUNDS
# -----------------------------
# Choose controllable variables and reasonable bounds. Edit as needed.
CONTROL_VARS = [
    "EnOxFl",      # Enriching oxygen flow
    "CoInSeVa",    # Coal injection set value
    "HoBlTe"       # Hot blast temperature
]

# Example bounds: (min, max) - replace with realistic ranges
BOUNDS = {
    "EnOxFl": (1000.0, 8000.0),
    "CoInSeVa": (0.0, 700.0),
    "HoBlTe": (800.0, 1400.0)
}



def prepare_input_row(baseline_row, control_values):
    """Return a DataFrame 1xN that contains the full feature vector for model
    prediction. baseline_row is a pandas Series representing default values for
    all features; control_values is a dict mapping control var -> value.
    """
    row = baseline_row.copy()
    for k, v in control_values.items():
        row[k] = v
    return pd.DataFrame([row])


# -----------------------------
# Objective wrapper for BO
# -----------------------------

def make_objective(model, baseline_row, feature_order):
    """
    Returns an objective function that BO will maximize. We want to minimize
    predicted SI, so return negative SI for maximization.
    """
    def objective(**kwargs):
        # kwargs will contain control var values
        input_df = prepare_input_row(baseline_row, kwargs)
        # Ensure columns align with model training order
        input_df = input_df[feature_order]
        pred = model.predict(input_df)[0]
        # Return negative because BayesianOptimization maximizes
        return -float(pred)

    return objective


# -----------------------------
# Feasibility projection via PuLP
# -----------------------------

def project_to_feasible(suggested_point, bounds, linear_constraints, baseline_row):
    """
    Formulate an LP that finds a feasible point closest to suggested_point by
    minimizing sum of absolute deviations. The LP variables are the control
    variables, subject to bounds and provided linear constraints.

    linear_constraints: list of tuples (coeff_dict, sense, rhs) where coeff_dict
    maps variable name to coefficient. RHS may depend on baseline_row and will
    typically be numeric.
    """
    prob = pulp.LpProblem("feasible_projection", pulp.LpMinimize)

    # Decision variables
    var_dict = {}
    for v in CONTROL_VARS:
        low, high = bounds[v]
        var_dict[v] = pulp.LpVariable(v, lowBound=low, upBound=high, cat='Continuous')

    # Auxiliary variables for absolute deviations
    pos_dev = {}
    neg_dev = {}
    for v in CONTROL_VARS:
        pos_dev[v] = pulp.LpVariable(f"pos_{v}", lowBound=0, cat='Continuous')
        neg_dev[v] = pulp.LpVariable(f"neg_{v}", lowBound=0, cat='Continuous')
        # deviation representation: var - suggested = pos - neg
        prob += var_dict[v] - suggested_point[v] == pos_dev[v] - neg_dev[v]

    # Objective: minimize sum of pos + neg deviations (L1 distance)
    prob += pulp.lpSum([pos_dev[v] + neg_dev[v] for v in CONTROL_VARS])

    # Add linear constraints
    for (coeffs, sense, rhs) in linear_constraints:
        expr = pulp.lpSum([coeffs.get(v, 0.0) * var_dict[v] for v in CONTROL_VARS if v in coeffs])
        if sense == '<=':
            prob += expr <= rhs
        elif sense == '>=':
            prob += expr >= rhs
        elif sense == '==':
            prob += expr == rhs

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    feasible_point = {v: var_dict[v].varValue for v in CONTROL_VARS}
    return feasible_point


# -----------------------------
# Main optimization routine
# -----------------------------

def optimize_with_bayes(model, baseline_row, feature_order, bounds, n_iter=25, linear_constraints=None):
    # Prepare pbounds for BayesianOptimization
    pbounds = {v: bounds[v] for v in CONTROL_VARS}

    # Create BO objective
    objective = make_objective(model, baseline_row, feature_order)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    # Run optimization
    optimizer.maximize(init_points=5, n_iter=n_iter)

    # Best suggested point (maximize negative SI -> minimize SI)
    best = optimizer.max['params']
    best_score = -optimizer.max['target']  # predicted SI

    # Prepare linear constraints (instantiate RHS if callable)
    instantiated_constraints = []
    if linear_constraints:
        for coeffs, sense, rhs in linear_constraints:
            if callable(rhs):
                rhs_val = rhs(baseline_row)
            else:
                rhs_val = rhs
            instantiated_constraints.append((coeffs, sense, rhs_val))

    # Project suggested point to feasible set using PuLP
    feasible_point = project_to_feasible(best, bounds, instantiated_constraints, baseline_row)

    # Evaluate model prediction at feasible point
    input_df = prepare_input_row(baseline_row, feasible_point)
    input_df = input_df[feature_order]
    pred_si = model.predict(input_df)[0]

    result = {
        'bo_suggested': best,
        'bo_predicted_SI': best_score,
        'feasible_point': feasible_point,
        'feasible_predicted_SI': float(pred_si)
    }

    return result


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, default=25, help='Number of BO iterations')
    parser.add_argument('--out', default='opt_result.json', help='Output JSON file')
    parser.add_argument('--max_total_flow_multiplier', type=float, default=1.2,
                        help='Multiplier for baseline total flow when setting flow constraints')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Load model
    model = joblib.load("Models/SI_xgb_model.pkl")

    # Load input and take last row as baseline
    df_input = pd.read_csv("Data/sample_input.csv")
    baseline_row = df_input.dropna().iloc[-1]

    feature_order = list(df_input.columns)
    if 'SI' in feature_order:
        feature_order.remove('SI')

    # Set up example linear constraint: EnOxFl + CoBlFl <= multiplier * baseline_total_flow
    # If CoBlFl not present in CONTROL_VARS, include it via RHS only; otherwise constraint uses vars in CONTROL_VARS
    baseline_total_flow = 0.0
    if 'EnOxFl' in baseline_row and 'CoBlFl' in baseline_row:
        baseline_total_flow = float(baseline_row['EnOxFl'] + baseline_row['CoBlFl'])
    else:
        # Fallback to EnOxFl baseline only
        baseline_total_flow = float(baseline_row.get('EnOxFl', 0.0))

    max_total_flow = args.max_total_flow_multiplier * baseline_total_flow

    # Define linear constraints: try to respect total flow
    linear_constraints = [
        ({'EnOxFl': 1.0}, '<=', max_total_flow)
    ]

    # Run optimization
    result = optimize_with_bayes(model, baseline_row, feature_order, BOUNDS, n_iter=args.n_iter,
                                linear_constraints=linear_constraints)

    # Save results
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)

    print("Optimization finished. Results saved to", args.out)
