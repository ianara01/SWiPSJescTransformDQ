"""
Docstring for core.search.surrogate
bflow에서 low-prob 영역 skip 가능
"""
def train_surrogate(df):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    X = df[["awg","par","turns","rpm"]]
    y = df["margin_pct"]
    model.fit(X,y)
    return model

def mcmc_search(initial_point, model):
    current = initial_point
    for _ in range(1000):
        proposal = perturb(current)
        if accept(proposal, current, model):
            current = proposal
    return current
