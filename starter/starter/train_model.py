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
        dataset, test_size = 0.20, random_state = 42)
    return train_dataset, test_dataset


def process_train_dataset(train_dataset):
    X_train, y_train, encoder, lb = ml_data.process_data(
        train_dataset, categorical_features = CAT_FEATURES, 
        label = LABEL_FEATURE, training = True)
    print('Saving encoder:\n', encoder)
    ml_model.save_object(encoder, ENCODER_FILE)
    print('Saving lb:\n', lb)
    ml_model.save_object(lb, LB_FILE)
    return X_train, y_train, encoder, lb

# Proces the test data with the process_data function.

def process_test_dataset(test_dataset, encoder, lb):
    X_test, y_test, encoder, lb = ml_data.process_data(
        test_dataset, categorical_features = CAT_FEATURES, 
        label = LABEL_FEATURE, training = False, encoder = encoder, lb = lb)
    return X_test, y_test

def process_datasets(train_dataset, test_dataset):
    X_train, y_train, encoder, lb = process_train_dataset(train_dataset)
    X_test, y_test = process_test_dataset(test_dataset, encoder, lb)
    return X_train, y_train, X_test, y_test

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
    results = f'precision={precision}\nrecall={recall}\nfbeta={fbeta}\n'
    print("\n" + results)
    write_to_text_file(SLICE_TEXTFILE, results)
    
def print_dataset_info(name, dataset):
    print("{} {}:\n{}".format(name, dataset.shape, dataset.head(5)))
    
def print_xy(title, X, y):
    print("{} {}:\n{}".format(title, X.shape, X[:5]))

def main():
    dataset = load_dataset(DATA_FILE)
    print_dataset_info("\nWhole dataset", dataset)
    train_dataset, test_dataset = split_dataset(dataset)
    print_dataset_info("\nTrain dataset", train_dataset)
    print_dataset_info("\nTest dataset", test_dataset)
    X_train, y_train, X_test, y_test = process_datasets(train_dataset, test_dataset)
    #print_xy('\nX_train', X_train, y_train)
    print('\nX_test[0]:\n', X_test[0])
    
    train_and_save_model(X_train, y_train)
    
    model = ml_model.load_model(MODEL_FILE)
    compute_results(model, X_test, y_test)
    
if __name__ == "__main__":
    main()
