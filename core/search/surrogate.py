"""
Docstring for core.search.surrogate
bflow에서 low-prob 영역 skip 가능
"""

# core/search/surrogate.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_surrogate(df):
    model = RandomForestRegressor(n_estimators=50)
    X = df[["AWG","Parallels","Turns_per_slot_side","rpm"]]
    y = df["margin_pct"]
    model.fit(X,y)
    return model

def predict_margin(model, params):
    return model.predict([params])[0]

def perturb(point):
    awg, par, turns, rpm = point
    return (
        awg + np.random.randint(-1,2),
        par + np.random.randint(-2,3),
        turns + np.random.randint(-2,3),
        rpm
    )

def accept(proposal, current, model):
    p_margin = predict_margin(model, proposal)
    c_margin = predict_margin(model, current)

    if p_margin > c_margin:
        return True

    prob = np.exp((p_margin - c_margin)/5.0)
    return np.random.rand() < prob

def mcmc_search(initial_point, model, steps=500):
    current = initial_point

    for _ in range(steps):
        proposal = perturb(current)
        if accept(proposal, current, model):
            current = proposal

    return current
