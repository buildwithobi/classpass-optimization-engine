import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

class BookingLikelihoodModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators = 200,
            max_depth = 6,
            learning_rate = 00.05,
            subsample = 0.8,
            colsample_bytree = 0.8
        )
    def prepare_features(self, df):
        df["hour"] = pd.to_datetime(df["class_time"]).dt.hour
        df["day_of_week"] = pd.to_datetime(df["class_time"]).dt.dayofweek
        df["streak"] = df["current_streak"] 
        df["social_score"] = df["social_score"]

        features = [
            "hour",
            "day_of_week",
            "streak",
            "social_score",
            "credits_remaining",
            "past_bookings_30d"
        ]
        
        return df[features], df["booked"]
    def train(self,df): 
        X, y = self.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.2, random_state = 42
        )

        self.model.fit(X_train, y_train)
        preds = self.model.predict_proba(X_test) [:, 1]
        print("AUC:", roc_auc_score(y_test, preds))

    def predict(self, user_features):
        return self.model.predict_proba(user_features) [:, 1]

