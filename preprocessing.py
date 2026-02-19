import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# one-hot encoding for categorical variables

def preprocess_features(df, target = None):
   
    # copyying to avoid modifying original
    df = df.copy()

    # one-hot encoding for categorical variables
    encoder = OneHotEncoder(drop = "first", sparse_output = False)
    class_encoded = encoder.fit_transform(df[["class_type"]])
    class_df = pd.DataFrame(class_encoded, columns = encoder.get_feature_names_out(["class_type"]))
    df = pd.concat([df.drop("class_type", axis = 1), class_df], axis = 1)

    #scale numerical features
    numeric_cols = ["credit_price", "current_streak", "social_score", "days_since_last_booking"]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    #split target
    if target:
        y = df[target]
        X = df.drop(columns = [target])
        return X, y
    else:
        return df


