import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# one-hot encoding for categorical variables

def preprocess_features(df, target = None):
   
    # copying to avoid modifying original
    df = df.copy()

    # one-hot encoding for categorical variables

    categorical_cols = ["class_type", "persona"]

    encoder = OneHotEncoder(drop = "first", sparse_output = False)
    class_encoded = encoder.fit_transform(df[categorical_cols])

    class_encoded_df = pd.DataFrame(class_encoded, columns = encoder.get_feature_names_out(categorical_cols),
    index = df.index)

    df = pd.concat([df.drop(columns = categorical_cols), class_encoded_df], axis = 1)

    #scale numerical features
    numeric_cols = ["credit_price", "current_streak", "social_score", "days_since_last_booking"]
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df = df.select_dtypes(include = ["int64", "float64", "bool"])

    #split target
    if target:
        y = df[target]
        X = df.drop(columns = [target])
        return X, y
    else:
        return df


