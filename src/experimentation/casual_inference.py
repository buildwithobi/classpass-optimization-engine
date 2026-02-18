import statsmodels.api as sm

def estimate_price_elasticity(df):
    X = df[["credit_price", "current_price", "social_score"]]
    X = sm.add_constant(X)
    y = df["booked"]

    model = sm.Logit(y, X).fit()
    return model.summary()