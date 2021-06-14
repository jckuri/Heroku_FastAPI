# Put the code for your API here.

import fastapi
import pydantic
import pandas
import numpy

import starter.starter.ml.model as ml_model
import starter.starter.train_model as train_model

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("pip install 'dvc[s3]'")
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = fastapi.FastAPI()


@app.get("/")
def read_root():
    return "Hello world"


DF_COLUMNS = [
    'age',
    'workclass',
    'fnlgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'salary']


class ModelFunctions:

    def __init__(self):
        self.encoder = ml_model.load_object(train_model.ENCODER_FILE)
        self.lb = ml_model.load_object(train_model.LB_FILE)
        self.model = ml_model.load_model(train_model.MODEL_FILE)

    def person_to_df(self, p):
        data = [[p.age, p.workclass, p.fnlgt, p.education, p.education_num,
                 p.marital_status, p.occupation, p.relationship, p.race, p.sex,
                 p.capital_gain, p.capital_loss, p.hours_per_week,
                 p.native_country, '<=50K']]
        df = pandas.DataFrame(data, columns=DF_COLUMNS)
        return df

    def process_row(self, df):
        x, y = train_model.process_test_dataset(df, self.encoder, self.lb)
        return x

    def person_to_numpy(self, p):
        df = self.person_to_df(p)
        x = self.process_row(df)
        x = x.astype(numpy.float)
        return x


mf = ModelFunctions()


class Person(pydantic.BaseModel):
    age: int
    workclass: str
    fnlgt: str
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


def get_person_1():
    desc = "Person 1. Her predicted salary should be 0, " \
        "which means she earns less than $50K."
    return {
        "summary": "Person 1",
        "description": desc,
        "value": {
            'age': 27,
            'workclass': 'Private',
            'fnlgt': 160178,
            'education': 'Some-college',
            'education_num': 10,
            'marital_status': 'Divorced',
            'occupation': 'Adm-clerical',
            'relationship': 'Not-in-family',
            'race': 'White',
            'sex': 'Female',
            'capital_gain': 0,
            'capital_loss': 0,
            'hours_per_week': 38,
            'native_country': 'United-States'
        }
    }


def get_person_2():
    desc = "Person 2. His predicted salary should be 1, " \
        "which means he earns more than $50K."
    return {
        "summary": "Person 2",
        "description": desc,
        "value": {
            'age': 29,
            'workclass': 'Private',
            'fnlgt': 185908,
            'education': 'Bachelors',
            'education_num': 13,
            'marital_status': 'Married-civ-spouse',
            'occupation': 'Exec-managerial',
            'relationship': 'Husband',
            'race': 'Black',
            'sex': 'Male',
            'capital_gain': 0,
            'capital_loss': 0,
            'hours_per_week': 55,
            'native_country': 'United-States'
        }
    }


def get_examples_of_persons():
    two_examples = {
        "person1": get_person_1(),
        "person2": get_person_2()
    }
    return fastapi.Body(..., examples=two_examples)


@app.post('/predict_salary')
async def predict_salary(person: Person = get_examples_of_persons()):
    x = mf.person_to_numpy(person)
    print("\nx:\n", x)
    pred = mf.model.predict(x)
    print('\npred:\n', type(pred), pred.shape, pred)
    return int(pred[0])
