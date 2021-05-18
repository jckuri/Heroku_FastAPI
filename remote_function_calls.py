# Write a script that POSTS to the API using the requests module and returns 
# both the result of model inference and the status code. Include a screenshot
# of the result. Name this live_post.png.

import requests
import pydantic
import main as m

DEFAULT_URL = 'https://udacity-salary-predictor.herokuapp.com/'

#>>> r = requests.post("http://bugs.python.org", data={'number': 12524, 'type': 'issue', 'action': 'show'})
#>>> print(r.status_code, r.reason)
#200 OK
#>>> print(r.text[:300] + '...')

def post(url, data_dict):
    r = requests.post(url, data = data_dict)
    return r.status_code, r.reason, r.text

"""    
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
"""

def inference_post(person, url = DEFAULT_URL):
    url2 = url + 'predict_salary'
    #assert type(person) == pydantic.BaseModel
    data_dict = person.dict()
    print('data_dict', data_dict)
    status_code, reason, text = post(url2, data_dict)
    print('status code', status_code, reason)
    print('text:\n', text)

"""
age, workclass, fnlgt, education, education_num, marital_status,
occupation, relationship, race, sex, capital_gain, capital_loss,
hours_per_week, native_country
"""

def main():
    person = m.Person(age = 27, workclass = 'Private', fnlgt = 160178, 
        education = 'Some-college', education_num = 10, 
        marital_status = 'Divorced', occupation = 'Adm-clerical', 
        relationship = 'Not-in-family', race = 'White', sex = 'Female', 
        capital_gain = 0, capital_loss = 0, hours_per_week = 38, 
        native_country = 'United-States')    
    inference_post(person)
    
if __name__ == "__main__":
    main()
