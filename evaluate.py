from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

def evaluate_model(model, X_test, y_test):

    # evaluate classification model, model will return as dictionary of metrics

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_test, y_pred_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }
    return metrics