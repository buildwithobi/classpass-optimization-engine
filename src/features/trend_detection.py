import pandas as pd

def detect_trends(df):
    trend = (
        df.groupby("class_type")
        .size()
        .sort_values(ascending = False)
        .reset_index(name = "bookings")
    )
    trend["trend_score"] = trend["bookings"] / trend["bookings"].sum()
    return trend
