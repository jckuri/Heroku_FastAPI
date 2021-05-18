import pandas
import starter.starter.ml.data as ml_data
import starter.starter.ml.model as ml_model
import starter.starter.train_model as train_model


def test_load_dataset():
    df = train_model.load_dataset(train_model.DATA_FILE)
    assert type(df) == pandas.DataFrame
    assert df.shape == (32561, 15)


def test_split_dataset():
    df = train_model.load_dataset(train_model.DATA_FILE)
    train_dataset, test_dataset = train_model.split_dataset(df)
    assert type(train_dataset) == pandas.DataFrame
    assert type(test_dataset) == pandas.DataFrame
    assert train_dataset.shape == (26048, 15)
    assert test_dataset.shape == (6513, 15)

