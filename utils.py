import joblib
import os

def save_model(model, path):

    #creating folder
    os.makedirs(os.path.dirname(path), exist_ok = True)
    joblib.dump(model, path)
    
    # saving to disk
    joblib.dump(model, path)

def load_model(path):
    # loading model from disk
    return joblib.load(path)