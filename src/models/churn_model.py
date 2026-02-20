from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class ChurnModel:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, df):
        features = df[[
            "days_since_last_booking",
            "current_streak",
            "credits_remaining",
            "past_bookings_30d"

        ]]
        target = df["churned"]

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 42)

        self.model.fit(X_train, y_train)
        preds = self.model.predict_proba(X_test) [:, 1]
        print("Churn AUC:", roc_auc_score(y_test, preds))

    def predict(self, df):
        features = df[[
            "days_since_last_booking",
            "current_streak",
            "credits_remaining",
            "past_bookings_30d",
        ]]
        return self.model.predict_proba(features)[:, 1]