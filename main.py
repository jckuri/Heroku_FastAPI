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


@app.post('/predict_salary')
async def predict_salary(person: Person):
    x = mf.person_to_numpy(person)
    print("\nx:\n", x)
    pred = mf.model.predict(x)
    print('\npred:\n', type(pred), pred.shape, pred)
    return int(pred[0])
