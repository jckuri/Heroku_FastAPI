import numpy
import pandas
import starter.starter.ml.model as ml_model
import starter.starter.train_model as train_model


def test_load_dataset():
    df = train_model.load_dataset(train_model.DATA_FILE)
    assert isinstance(df, pandas.DataFrame)
    assert df.shape == (32561, 15)


def test_split_dataset():
    df = train_model.load_dataset(train_model.DATA_FILE)
    train_dataset, test_dataset = train_model.split_dataset(df)
    assert isinstance(train_dataset, pandas.DataFrame)
    assert isinstance(test_dataset, pandas.DataFrame)
    assert train_dataset.shape == (26048, 15)
    assert test_dataset.shape == (6513, 15)


def test_process_train_dataset():
    df = train_model.load_dataset(train_model.DATA_FILE)
    train_dataset, test_dataset = train_model.split_dataset(df)
    X_train, y_train, encoder, lb = train_model.process_train_dataset(
        train_dataset)
    assert isinstance(X_train, numpy.ndarray)
    assert isinstance(y_train, numpy.ndarray)
    assert X_train.shape == (26048, 108)
    assert y_train.shape == (26048,)


def test_process_test_dataset():
    df = train_model.load_dataset(train_model.DATA_FILE)
    train_dataset, test_dataset = train_model.split_dataset(df)
    X_train, y_train, encoder, lb = train_model.process_train_dataset(
        train_dataset)
    X_test, y_test = train_model.process_test_dataset(
        test_dataset, encoder, lb)
    assert isinstance(X_test, numpy.ndarray)
    assert isinstance(y_test, numpy.ndarray)
    assert X_test.shape == (6513, 108)
    assert y_test.shape == (6513,)


def test_seconds_to_string():
    assert train_model.seconds_to_string(53) == '53.00 seconds'
    assert train_model.seconds_to_string(128) == '2 minutes 8.00 seconds'
    assert train_model.seconds_to_string(128.231) == '2 minutes 8.23 seconds'


def test_inference():
    df = train_model.load_dataset(train_model.DATA_FILE)
    train_dataset, test_dataset = train_model.split_dataset(df)
    X_train, y_train, X_test, y_test, encoder, lb = \
        train_model.process_datasets(train_dataset, test_dataset)
    model = ml_model.load_model(train_model.MODEL_FILE)
    preds = ml_model.inference(model, X_test)
    assert isinstance(preds, numpy.ndarray)
    assert preds.shape == (6513,)


def test_compute_model_metrics():
    df = train_model.load_dataset(train_model.DATA_FILE)
    train_dataset, test_dataset = train_model.split_dataset(df)
    X_train, y_train, X_test, y_test, encoder, lb = \
        train_model.process_datasets(train_dataset, test_dataset)
    model = ml_model.load_model(train_model.MODEL_FILE)
    preds = ml_model.inference(model, X_test)
    precision, recall, fbeta = ml_model.compute_model_metrics(y_test, preds)
    assert isinstance(precision, numpy.float64)
    assert isinstance(recall, numpy.float64)
    assert isinstance(fbeta, numpy.float64)
