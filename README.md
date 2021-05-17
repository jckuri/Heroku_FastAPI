![Badge](https://github.com/jckuri/Heroku_FastAPI/actions/workflows/python-package.yml/badge.svg)

# Deploying a Machine Learning Model on Heroku with FastAPI

**Machine Learning DevOps Engineer Nanodegree<br/>
https://classroom.udacity.com/nanodegrees/nd0821**

# Usage and Quick Review

You can visit the link https://udacity-salary-predictor.herokuapp.com/docs
in which the web app is deployed.

The easiest way to test this web app is through the curl command. 
Here is the script `test_remote_api.sh` to test this web app:

```
echo "Testing GET:"
curl -X GET "https://udacity-salary-predictor.herokuapp.com"

#       age         workclass   fnlgt     education  education-num      marital-status         occupation   relationship   race     sex  capital-gain  capital-loss  hours-per-week native-country salary
#14160   27           Private  160178  Some-college             10            Divorced       Adm-clerical  Not-in-family  White  Female             0             0              38  United-States  <=50K   
#28868   29           Private  185908     Bachelors             13  Married-civ-spouse    Exec-managerial        Husband  Black    Male             0             0              55  United-States   >50K 

echo "\nResult of POST 1: "
curl -X POST "https://udacity-salary-predictor.herokuapp.com/predict_salary" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"age\":27, \"workclass\":\"Private\", \"fnlgt\":160178, \"education\":\"Some-college\", \"education_num\":10, \"marital_status\":\"Divorced\", \"occupation\":\"Adm-clerical\", \"relationship\":\"Not-in-family\", \"race\":\"White\", \"sex\":\"Female\", \"capital_gain\":0, \"capital_loss\":0, \"hours_per_week\":38, \"native_country\":\"United-States\"}"

echo "\nResult of POST 2: "
curl -X POST "https://udacity-salary-predictor.herokuapp.com/predict_salary" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"age\":29, \"workclass\":\"Private\", \"fnlgt\":185908, \"education\":\"Bachelors\", \"education_num\":13, \"marital_status\":\"Married-civ-spouse\", \"occupation\":\"Exec-managerial\", \"relationship\":\"Husband\", \"race\":\"Black\", \"sex\":\"Male\", \"capital_gain\":0, \"capital_loss\":0, \"hours_per_week\":55, \"native_country\":\"United-States\"}"

echo ""
```

And if you run the script `test_remote_api.sh`, you will get:

```
$ sh test_remote_api.sh 
Testing GET:
"Hello world"
Result of POST 1: 
0
Result of POST 2: 
1
```

What does it mean?
It means that you are calling the GET `/` and the result is a greeting: `"Hello world"`
You are also calling the POST `/predict_salary` twice by passing
2 registers of the dataset:

```
# REGISTER 1:
#       age         workclass   fnlgt     education  education-num      marital-status         occupation   relationship   race     sex  capital-gain  capital-loss  hours-per-week native-country salary
#14160   27           Private  160178  Some-college             10            Divorced       Adm-clerical  Not-in-family  White  Female             0             0              38  United-States  <=50K   
```

```
# REGISTER 2:
#       age         workclass   fnlgt     education  education-num      marital-status         occupation   relationship   race     sex  capital-gain  capital-loss  hours-per-week native-country salary
#28868   29           Private  185908     Bachelors             13  Married-civ-spouse    Exec-managerial        Husband  Black    Male             0             0              55  United-States   >50K 
```

`Result of POST 1` is `0` which means the predicted salary is `<=50K`.<br/>
`Result of POST 2` is `1` which means the predicted salary is `>50K`.<br/>
Both predictions are correct because they coincide with the salaries of register 1 and register 2.


# Environment Set up

Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.

## Repositories
* Create a directory for the project and initialize git and dvc.
    * As you work on the code, continually commit changes. Generated models you want to keep must be committed to dvc.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.
* Set up a remote repository for dvc.

# Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.
* Commit this modified data to dvc (we often want to keep the raw data untouched but then can keep updating the cooked version).

# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
   	 * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment
* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
    * Enable automatic deployments that only deploy if your continuous integration passes.
    * Hint: think about how paths will differ in your local environment vs. on Heroku.
    * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Write a script that uses the requests module to do one POST on your live API.

