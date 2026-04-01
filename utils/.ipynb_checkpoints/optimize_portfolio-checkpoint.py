from scipy.optimize import minimize
import numpy as np


def risk_budget_objective(weights, covar_matrix):
    """Objective function for equal risk contribution optimization."""
    weights = np.array(weights)
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(covar_matrix, weights)))
    marginal_cont = np.dot(covar_matrix, weights)/portfolio_vol
    cont = weights*marginal_cont
    target_cont = portfolio_vol / len(weights)
    return np.sum((cont - target_cont)**2)


def optimize_risk_parity(covar_matrix):
    """Optimize for equal risk contribution weights."""
    n_assets = len(covar_matrix)
    x0 = np.ones(n_assets)/n_assets
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    bounds = [(0.001, 0.999) for _ in range(n_assets)]
    result = minimize(
        risk_budget_objective,
        x0,
        args=(covar_matrix,),
        method = "SLSQP",
        bounds = bounds,
        constraints = constraints,
        options={"ftol": 1e-12, "disp": False},
    )
    return result.x


def calculate_risk_contributions(weights, covar_matrix):
    """Return each asset's fractional risk contribution."""
    weights = np.array(weights)
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(covar_matrix, weights)))
    marginal_cont = np.dot(covar_matrix, weights)/portfolio_vol
    cont = weights*marginal_cont
    return cont/portfolio_vol
