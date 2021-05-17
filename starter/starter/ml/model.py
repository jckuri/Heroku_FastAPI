import sklearn.ensemble
import sklearn.metrics
import joblib

# precision=0.781, recall=0.618, fbeta=0.690

# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    rfc = sklearn.ensemble.RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto'],  # 'sqrt'
        'max_depth': [25, 100],  # [5, 25, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = sklearn.model_selection.GridSearchCV(
        estimator=rfc, param_grid=param_grid, cv=5, verbose=4)
    cv_rfc.fit(X_train, y_train)
    return cv_rfc.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = sklearn.metrics.fbeta_score(y, preds, beta=1, zero_division=1)
    precision = sklearn.metrics.precision_score(y, preds, zero_division=1)
    recall = sklearn.metrics.recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(rfc, model_file):
    joblib.dump(rfc, model_file)


def load_model(model_file):
    rfc = joblib.load(model_file)
    return rfc


def save_object(obj, obj_file):
    joblib.dump(obj, obj_file)


def load_object(obj_file):
    obj = joblib.load(obj_file)
    return obj
