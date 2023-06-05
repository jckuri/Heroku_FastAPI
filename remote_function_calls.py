# Write a script that POSTS to the API using the requests module and returns
# both the result of model inference and the status code. Include a screenshot
# of the result. Name this live_post.png.

import requests
import main as m

#DEFAULT_URL = 'https://udacity-salary-predictor.herokuapp.com'
DEFAULT_URL = 'http://127.0.0.1:8000'

PERSON1 = m.Person(age=27, workclass='Private', fnlgt=160178,
                   education='Some-college', education_num=10,
                   marital_status='Divorced', occupation='Adm-clerical',
                   relationship='Not-in-family', race='White', sex='Female',
                   capital_gain=0, capital_loss=0, hours_per_week=38,
                   native_country='United-States')

PERSON2 = m.Person(
    age=29,
    workclass='Private',
    fnlgt=185908,
    education='Bachelors',
    education_num=13,
    marital_status='Married-civ-spouse',
    occupation='Exec-managerial',
    relationship='Husband',
    race='Black',
    sex='Male',
    capital_gain=0,
    capital_loss=0,
    hours_per_week=55,
    native_country='United-States')


# >>> r = requests.post("http://bugs.python.org",
#         data={'number': 12524, 'type': 'issue', 'action': 'show'})
# >>> print(r.status_code, r.reason)
# 200 OK
# >>> print(r.text[:300] + '...')

def post(url, data):
    r = requests.post(url, data=data)
    return r.status_code, r.reason, r.text


def get(url):
    r = requests.get(url)
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


def text_to_int(text, default_value):
    try:
        return int(text)
    except BaseException:
        return default_value


def inference_post(person, url=DEFAULT_URL):
    url2 = url + '/predict_salary'
    assert isinstance(person, m.Person)
    data = person.json()
    print('Input data:', data)
    status_code, reason, text = post(url2, data)
    print('Status code:', status_code, reason)
    if status_code == 200:
        result = text_to_int(text, default_value=-1)
        print('Result:', result)
        return status_code, result
    return status_code, reason


def root_get(url=DEFAULT_URL):
    url2 = url + '/'
    status_code, reason, text = get(url2)
    print('Status code:', status_code, reason)
    if status_code == 200:
        print('Result:', text)
        return status_code, text
    return status_code, reason


def main():
    print('\nGET /')
    status_code, result = root_get()
    print('\nPOST /predict_salary')
    status_code, result = inference_post(PERSON1)
    print('\nPOST /predict_salary')
    status_code, result = inference_post(PERSON2)


if __name__ == "__main__":
    main()
