import json

import requests

URL = "http://127.0.0.1:8000"

# TODO: send a GET using the URL http://127.0.0.1:8000
# TODO: print the status code
# TODO: print the welcome message
def get_request():
    print('\nGET request')
    PARAMS = {}
    r = requests.get(url = URL, params = PARAMS)
    print(f'Status code: {r}')
    print(f'Welcome message: {r.text}')


data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

data2 = {
    "age": 27, 
    "workclass": "Private", 
    "fnlgt": 160178, 
    "education": "Some-college", 
    "education_num": 10, 
    "marital_status": "Divorced", 
    "occupation": "Adm-clerical", 
    "relationship": "Not-in-family", 
    "race": "White", 
    "sex": "Female", 
    "capital_gain": 0, 
    "capital_loss": 0, 
    "hours_per_week": 38, 
    "native_country": "United-States"
}

data3 = {
    "age":29, 
    "workclass":"Private", 
    "fnlgt":185908, 
    "education":"Bachelors", 
    "education_num":13, 
    "marital_status":"Married-civ-spouse", 
    "occupation":"Exec-managerial", 
    "relationship":"Husband", 
    "race":"Black", 
    "sex":"Male", 
    "capital_gain":0, 
    "capital_loss":0, 
    "hours_per_week":55, 
    "native_country":"United-States"
}


# TODO: send a POST using the data above
# TODO: print the status code
# TODO: print the result
# https://www.geeksforgeeks.org/python-requests-post-request-with-headers-and-body/
def post_request(data):
    print('\nPOST request. Data:')
    print(data)
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    r = requests.post(url = URL + "/predict_salary", headers = headers, json = data)
    print(f'Status code: {r}')
    print(f'Result: "{r.text}"')


def main():
    get_request()
    post_request(data2)
    post_request(data3)
    print()


main()
