import numpy as np
import pandas as pd

CLASS_TYPES = ["Lagree", "HIIT", "Reformer Pilates", "Heated Mat Pilates", "Yoga Sculpt", "Boxing"]

PERSONAS = {
    "Lagree Loyalist": {"preferred_class": "Lagree", "base_logit": 0.8, "social_weight": 0.6, "streak_weight": 0.8},
    "HIIT Hustler": {"preferred_class": "HIIT", "base_logit": 0.4, "social_weight": 0.3, "streak_weight": 0.7},
    "Pilates Princess": {"preferred_class": "Reformer Pilates", "base_logit": 1.0, "social_weight": 0.5, "streak_weight": 0.5},
    "Balanced Burner": {"preferred_class": "Heated Mat Pilates", "base_logit": 0.5, "social_weight": 0.3, "streak_weight": 0.5},
    "Mindful Muse": {"preferred_class": "Yoga Sculpt", "base_logit": 0.6, "social_weight": 0.4, "streak_weight": 0.6},
    "Boxing Baddie": {"preferred_class": "Boxing", "base_logit": 0.3, "social_weight": 0.5, "streak_weight": 0.9} 
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_classpass_data(n = 50000):
    np.random.seed(42)

    data = pd.DataFrame({
        "persona": np.random.choice(list(PERSONAS.keys()), n),
        "class_type": np.random.choice(CLASS_TYPES, n),
        "credits_remaining": np.random.randint(0, 50, n),
        "credit_price": np.random.randint(6, 21, n),
        "current_streak": np.random.randint(0, 100, n),
        "past_bookings_30d": np.random.randint(0, 10, n),
        "social_score": np.random.rand(n) * 100,
        "days_since_last_booking": np.random.randint(0, 30, n),
        "class_time": pd.date_range(start = "2026-01-01", periods = n, freq = "h")
    })
        
    # add churn label with some random noise
    data["churned"] = np.random.choice([0, 1], size = len(data))

    return data
    
    # defined probability of booking based on persona and features
    def booking_probability(row):
        persona = PERSONAS[row["persona"]]

        base = persona["base_logit"]

        preference_bonus = 0.8 if row["class_type"] == persona["preferred_class"] else 0
        
        streak_effect = persona["streak_weight"] * (row["current_streak"] / 100)
        
        social_effect = persona["social_weight"] * row["social_score"]
        
        price_penalty = -0.08 * row["credit_price"]

        inactivity_penalty = -0.05 * row["days_since_last_booking"]
        
        logit = (
            base
            + preference_bonus 
            + streak_effect
            + social_effect
            + price_penalty
            + inactivity_penalty
        )
        return 1 / (1 + np.exp(-logit))
    
    data["prob"] = data.apply(booking_probability, axis = 1)
    data["booked"] = np.random.binomial(1, data["prob"])

    return data
