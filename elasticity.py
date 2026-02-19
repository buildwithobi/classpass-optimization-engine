import numpy as np

def estimate_price_effect(model, X, feature_name = "credit_price"):
    # Estimate price elasticity by calculating the marginal effect of price on booking probability.
    #model: trained booking likelihood model
    # X: feature matrix
    # feature_name: name of the price feature to analyze (e.g. "current_price")
    
   if hasattr(model, "coef_"):
    coef_index = list(X.columns).index(feature_name)
    return model.coef_[0][coef_index]
    else:
        raise ValueError("Model does not have coef_ attribute. Ensure it's a linear model.")