# Put the code for your API here.

import fastapi
import pydantic
import pandas
import numpy

import starter.starter.ml.model as ml_model
import starter.starter.ml.data as ml_data
import starter.starter.train_model as train_model

app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return "Hello world"
    
"""
       age         workclass   fnlgt     education  education-num      marital-status         occupation   relationship   race     sex  capital-gain  capital-loss  hours-per-week native-country salary
14160   27           Private  160178  Some-college             10            Divorced       Adm-clerical  Not-in-family  White  Female             0             0              38  United-States  <=50K
27048   45         State-gov   50567       HS-grad              9  Married-civ-spouse    Exec-managerial           Wife  White  Female             0             0              40  United-States  <=50K
28868   29           Private  185908     Bachelors             13  Married-civ-spouse    Exec-managerial        Husband  Black    Male             0             0              55  United-States   >50K
5667    30           Private  190040     Bachelors             13       Never-married  Machine-op-inspct  Not-in-family  White  Female             0             0              40  United-States  <=50K
7827    29  Self-emp-not-inc  189346  Some-college             10            Divorced       Craft-repair  Not-in-family  White    Male          2202             0              50  United-States  <=50K
...
"""    

DF_COLUMNS = ['age', 'workclass', 'fnlgt', 'education', 'education-num', 
    'marital-status', 'occupation', 'relationship', 'race', 'sex', 
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 
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
        df = pandas.DataFrame(data, columns = DF_COLUMNS)
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
    age:int
    workclass:str
    fnlgt:str
    education:str
    education_num:int
    marital_status:str
    occupation:str
    relationship:str
    race:str
    sex:str
    capital_gain:int
    capital_loss:int
    hours_per_week:int
    native_country:str
    
@app.post('/predict_salary')
async def predict_salary(person: Person):
    x = mf.person_to_numpy(person)
    print("\nx:\n", x)
    pred = mf.model.predict(x)
    print('\npred:\n', type(pred), pred.shape, pred)
    return int(pred[0])

#       age         workclass   fnlgt     education  education-num      marital-status         occupation   relationship   race     sex  capital-gain  capital-loss  hours-per-week native-country salary
#14160   27           Private  160178  Some-college             10            Divorced       Adm-clerical  Not-in-family  White  Female             0             0              38  United-States  <=50K   
#28868   29           Private  185908     Bachelors             13  Married-civ-spouse    Exec-managerial        Husband  Black    Male             0             0              55  United-States   >50K 
    
# curl -X POST "http://127.0.0.1:8000/predict_salary" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"age\":27, \"workclass\":\"Private\", \"fnlgt\":160178, \"education\":\"Some-college\", \"education_num\":10, \"marital_status\":\"Divorced\", \"occupation\":\"Adm-clerical\", \"relationship\":\"Not-in-family\", \"race\":\"White\", \"sex\":\"Female\", \"capital_gain\":0, \"capital_loss\":0, \"hours_per_week\":38, \"native_country\":\"United-States\"}"

# curl -X POST "http://127.0.0.1:8000/predict_salary" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"age\":29, \"workclass\":\"Private\", \"fnlgt\":185908, \"education\":\"Bachelors\", \"education_num\":13, \"marital_status\":\"Married-civ-spouse\", \"occupation\":\"Exec-managerial\", \"relationship\":\"Husband\", \"race\":\"Black\", \"sex\":\"Male\", \"capital_gain\":0, \"capital_loss\":0, \"hours_per_week\":55, \"native_country\":\"United-States\"}"
