import pandas as pd

# one-hot encoding for categorical variables

def preprocess_data(df):
    df = df.copy()

    df = pd.get_dummies(df, columns = ["persona", "class_type"], drop_first = True)

    return df


