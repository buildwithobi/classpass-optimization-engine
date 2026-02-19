from sklearn.linear_model import LogisticRegression

def train_logistic_model(X, y):
    model = LogisticRegression(max_iter = 1000)
    model.fit(X, Y)
    return model