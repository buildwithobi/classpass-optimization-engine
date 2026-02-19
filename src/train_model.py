from src.moodels.booking_prediction import BookingLikelihoodModel
from src.models.churn_model import ChurnModel

def train_booking_model(X_train, y_train):
    model = BookingLikelihoodModel()
    model.model.fit(X_train, y_train)
    return model

def train_churn_model(X_train, y_train):
    model = ChurnModel()
    model.model.fit(X_train, y_train)
    return model