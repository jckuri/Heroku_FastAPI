# Model Card

For additional information, see the Model Card paper:<br/>
https://arxiv.org/pdf/1810.03993.pdf

## Model Details

### Person developing model:
Juan Carlos Kuri Pinto

### Model date:
May 17, 2021.

### Model version:
1.5

### Model type:
Random Forest Classifier

### Information about training algorithms, parameters, fairness constraints or other applied approaches, and features:

**Learning algorithm:** Random Forest Classifier

**Parameters:**

```
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto'],
    'max_depth': [25, 100],
    'criterion': ['gini', 'entropy']
}
```

```
sklearn.model_selection.GridSearchCV
(estimator=Random Forests Classifier, param_grid = param_grid, cv = 5)
```

**Features:**<br/>
age: continuous. (Young <=50, Old >50)<br/>
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.<br/>
fnlwgt: continuous.<br/>
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.<br/>
education-num: continuous.<br/>
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.<br/>
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.<br/>
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.<br/>
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.<br/>
sex: Female, Male.<br/>
capital-gain: continuous.<br/>
capital-loss: continuous.<br/>
hours-per-week: continuous.<br/>
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.<br/>

**Label:**<br/>
salary: >50K, <=50K.

### Paper or other resource for more information:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

### Citation details:
https://archive.ics.uci.edu/ml/datasets/census+income

### License:
Donation.

### Contact to send questions or comments about the model:

**Model creator:**<br/>
Juan Carlos Kuri Pinto<br/>
jckuri@gmail.com

**Dataset donor:**<br/>
Ronny Kohavi and Barry Becker<br/>
Data Mining and Visualization<br/>
Silicon Graphics.<br/>
e-mail: ronnyk '@' sgi.com for questions.

--------------------------------------------------------------------------------

## Intended Use

### Primary intended uses

### Primary intended users

### Out-of-scope use cases

--------------------------------------------------------------------------------

## Factors

## Metrics

## Evaluation Data

## Training Data

## Quantitative Analyses

## Ethical Considerations

## Caveats and Recommendations
