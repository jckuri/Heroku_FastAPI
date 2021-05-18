# Script to train machine learning model.

import sklearn.model_selection

# Add the necessary imports for the starter code.

import pandas as pd
import time
import starter.starter.ml.data as ml_data
import starter.starter.ml.model as ml_model

DATA_FILE = "starter/data/census.csv"
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]
LABEL_FEATURE = "salary"
MODEL_FILE = 'starter/model/rfc_model.pkl'
ENCODER_FILE = 'starter/model/encoder.pkl'
LB_FILE = 'starter/model/lb.pkl'
SLICE_TEXTFILE = 'slice_output.txt'

# Add code to load in the data.


def load_dataset(data_file):
    dataset = pd.read_csv(data_file)
    return dataset


# Optional enhancement, use K-fold cross validation
# instead of a train-test split.

def split_dataset(dataset):
    train_dataset, test_dataset = sklearn.model_selection.train_test_split(
        dataset, test_size=0.20, random_state=42)
    return train_dataset, test_dataset


def process_train_dataset(train_dataset):
    X_train, y_train, encoder, lb = ml_data.process_data(
        train_dataset, categorical_features=CAT_FEATURES,
        label=LABEL_FEATURE, training=True)
    print('Saving encoder:\n', encoder)
    ml_model.save_object(encoder, ENCODER_FILE)
    print('Saving lb:\n', lb)
    ml_model.save_object(lb, LB_FILE)
    return X_train, y_train, encoder, lb

# Proces the test data with the process_data function.


def process_test_dataset(test_dataset, encoder, lb):
    X_test, y_test, encoder, lb = ml_data.process_data(
        test_dataset, categorical_features=CAT_FEATURES,
        label=LABEL_FEATURE, training=False, encoder=encoder, lb=lb)
    return X_test, y_test


def process_datasets(train_dataset, test_dataset):
    X_train, y_train, encoder, lb = process_train_dataset(train_dataset)
    X_test, y_test = process_test_dataset(test_dataset, encoder, lb)
    print(f'X_train.shape={X_train.shape}')
    print(f'y_train.shape={y_train.shape}')
    print(f'X_test.shape={X_test.shape}')
    print(f'y_test.shape={y_test.shape}')
    return X_train, y_train, X_test, y_test, encoder, lb


def seconds_to_string(seconds):
    m = int(seconds / 60)
    if m == 0:
        return f'{seconds:.2f} seconds'
    s = seconds - m * 60
    return f'{m} minutes {s:.2f} seconds'

# Train and save a model.


def train_and_save_model(X_train, y_train):
    print('\nTraining model...')
    t0 = time.time()
    model = ml_model.train_model(X_train, y_train)
    dt = time.time() - t0
    print(f'The model was trained in {seconds_to_string(dt)}.')
    ml_model.save_model(model, MODEL_FILE)


def write_to_text_file(textfile, text):
    with open(textfile, 'w') as f:
        f.write(text)
        f.close()


def compute_results(model, X_test, y_test):
    preds = ml_model.inference(model, X_test)
    precision, recall, fbeta = ml_model.compute_model_metrics(y_test, preds)
    f1_score = 2 * precision * recall / (precision + recall)
    results = f'precision={precision}\nrecall={recall}\n'
    results += f'fbeta={fbeta}\nf1_score={f1_score}\n'
    print("\n" + results)


def print_dataset_info(name, dataset):
    print("{} {}:\n{}".format(name, dataset.shape, dataset.head(5)))


def print_xy(title, X, y):
    print("{} {}:\n{}".format(title, X.shape, X[:5]))


def compute_results_of_slice(slice_df, model, encoder, lb):
    X, y = process_test_dataset(slice_df, encoder, lb)
    preds = ml_model.inference(model, X)
    precision, recall, fbeta = ml_model.compute_model_metrics(y, preds)
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, fbeta, f1_score


def classify_test_dataset(test_dataset, model, encoder, lb):
    young_people = test_dataset[test_dataset['age'] <= 50]
    old_people = test_dataset[test_dataset['age'] > 50]
    men = test_dataset[test_dataset['sex'] == 'Male']
    women = test_dataset[test_dataset['sex'] == 'Female']
    print('young_people.shape', young_people.shape)
    print('old_people.shape', old_people.shape)
    print(
        'young_people + old_people:',
        young_people.shape[0] +
        old_people.shape[0],
        '\n')
    print('men.shape', men.shape)
    print('women.shape', women.shape)
    print('men + women:', men.shape[0] + women.shape[0], '\n')
    young_men = young_people[young_people['sex'] == 'Male']
    young_women = young_people[young_people['sex'] == 'Female']
    old_men = old_people[old_people['sex'] == 'Male']
    old_women = old_people[old_people['sex'] == 'Female']
    print('young_men.shape', young_men.shape)
    print('young_women.shape', young_women.shape)
    print('old_men.shape', old_men.shape)
    print('old_women.shape', old_women.shape)
    print(
        'young_men + young_women + old_men + old_women:',
        young_men.shape[0] +
        young_women.shape[0] +
        old_men.shape[0] +
        old_women.shape[0],
        '\n')
    print('ALL: test_dataset.shape', test_dataset.shape)
    slices_dict = {
        'Young Men': young_men,
        'Young Women': young_women,
        'Old Men\t': old_men,
        'Old Women': old_women,
        'Young\t': young_people,
        'Old\t': old_people,
        'Men\t': men,
        'Women\t': women,
        'Test Dataset': test_dataset}
    report = '\nSLICE\t\t\tPRECISION\tRECALL\t\tF-BETA\t\tF1-SCORE\n'
    for slice_name in slices_dict.keys():
        slice_df = slices_dict[slice_name]
        precision, recall, fbeta, f1_score = compute_results_of_slice(
            slice_df, model, encoder, lb)
        report += '{}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\n'.format(
            slice_name, precision, recall, fbeta, f1_score)
    report += '\n'
    print(report)
    write_to_text_file(SLICE_TEXTFILE, report)


def main():
    dataset = load_dataset(DATA_FILE)
    print_dataset_info("\nWhole dataset", dataset)
    train_dataset, test_dataset = split_dataset(dataset)
    print_dataset_info("\nTrain dataset", train_dataset)
    print_dataset_info("\nTest dataset", test_dataset)
    X_train, y_train, X_test, y_test, encoder, lb = process_datasets(
        train_dataset, test_dataset)
    print('\nX_test[0]:\n', X_test[0])

    # train_and_save_model(X_train, y_train)

    model = ml_model.load_model(MODEL_FILE)
    compute_results(model, X_test, y_test)

    classify_test_dataset(test_dataset, model, encoder, lb)


if __name__ == "__main__":
    main()
