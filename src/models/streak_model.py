import pandas as pd

def calculate_streak(bookings):
    bookings = bookings.sort_values("date")
    bookings["prev_date"] = bookings["date"].shift(1)
    bookings["gap"] = (
        bookings["date"] - bookings["prev_date"]
    ).dt.days

    streak = 1
    for gap in bookings["gap"]:
        if gap == 1:
            streak += 1
        else:
            streak = 1
        
    return streak