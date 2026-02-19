import joblib

def save_model(model, path):
    # saving to disk
    joblib.dump(model, path)

def load_model(path):
    # loading model from disk
    return joblib.load(path)