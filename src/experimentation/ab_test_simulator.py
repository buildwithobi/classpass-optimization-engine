import numpy as np
from scipy import stats

def run_ab_test(control_rate, test_rate, n = 1000):
    control = np.random.binomial(1, control_rate, n)
    test = np.random.binomial(1, test_rate, n)

    stat, p_value = stats.ttest_ind(control, test)
    lift = test.mean() - control.mean()

    return{
        "control_conversion": control.mean(),
        "test_conversion": test.mean(),
        "lift": lift,
        "p_value": p_value
    }