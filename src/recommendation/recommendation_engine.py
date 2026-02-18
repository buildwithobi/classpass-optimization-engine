import numpy as np

class ClassRecommender:
    def __init__(self, booking_model):
        self.model = booking_model
    
    def recommend(self, user_features, class_options):
        scores = self.model.predict(class_options)
        class_options["score"] = scores
        return class_options.sort_values("score", ascending = False).head(5)
    