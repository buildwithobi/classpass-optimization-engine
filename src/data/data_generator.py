import numpy as np
import pandas as pd

CLASS_TYPES = ["Lagree", "HIIT", "Reformer Pilates", "Heated Mat Pilates", "Yoga Sculpt", "Boxing"]

PERSONAS = {
    "lagree_loyalist": {"preferred_class": "Lagree", "base_prob": 0.65, "social_weight": 0.5, "streak_weight": 0.6},
    "hiit_hustler": {"preferred_class": "HIIT", "base_prob": 0.55, "social_weight": 0.2, "streak_weight": 0.5},
    "pilates_princess": {"preferred_class": "Reformer Pilates", "base_prob": 0.75, "social_weight": 0.5, "streak_weight": 0.4},
    "balanced_burner": {"preferred_class": "Heated Mat Pilates", "base_prob": 0.58, "social_weight": 0.2, "streak_weight": 0.4},
    "mindful_muse": {"preferred_class": "Yoga Sculpt", "base_prob": 0.6, "social_weight": 0.3, "streak_weight": 0.5},
    "boxing_baddie": {"preferred_class": "Boxing", "base_prob": 0.5, "social_weight": 0.4, "streak_weight": 0.6} 
}

def generate_classpass_data(n = 50000):
    np.random.seed(42)

    data = pd.DataFrame({
        "persona": np.random.choice(list(PERSONAS.keys()), n),
        "class_type": np.random.choice(CLASS_TYPES, n),
        "credit_price": np.random.randint(6,21,n),
        "current_streak": np.random.randint(0, 366, n),
        "social_score": np.random.uniform(0, 1, n),
        "days_since_last_booking": np.random.randint(0, 30, n),
    })

    def booking_probability(row):
        persona = PERSONAS[row["persona"]]
        base = persona["base_prob"]

        preferrence_bonus = 0.1 if row["class_type"] -- persona["preferred_class"] else 0
        
        return(
            base
            + preferrence_bonus
            + persona["streak_weight"] * (row["current_streak"] / 100)
            + persona["social_weight"] * row["social_score"]
            - 0.02 * row["credit_price"]
        )
    
    data["prob"] = data.apply(booking_probability, axis = 1)
    data["booked"] = np.random.binomial(1, np.clip(data["prob"], 0, 1))

    return data
