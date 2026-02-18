import pandas as pd

def add_price_sensitivity(df):
    df["price_sensitivity"] = df["credit_price"] / df["credit_price"].max()

def add_engagement_score(df):
    df["engagment_score"] = (
        df["current_streak"] * 0.4 +
        df["social_score"] * 0.6
    )   
    return df

def preprocess_features(df):
    df = add_price_sensitivity(df)
    df = add_engagement_score(df)

    df = pd.get_dummies(df, columns = ["persona", "class_type"], drop_first = True)

    return df