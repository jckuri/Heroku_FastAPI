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

**FEATURES:**<br/>
**age:** continuous. (Young <=50, Old >50)<br/>
**workclass:** Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.<br/>
**fnlwgt:** continuous.<br/>
**education:** Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.<br/>
**education-num:** continuous.<br/>
**marital-status:** Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.<br/>
**occupation:** Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.<br/>
**relationship:** Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.<br/>
**race:** White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.<br/>
**sex:** Female, Male.<br/>
**capital-gain:** continuous.<br/>
**capital-loss:** continuous.<br/>
**hours-per-week:** continuous.<br/>
**native-country:** United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.<br/>

**LABEL:**<br/>
**salary:** >50K, <=50K.

### Paper or other resource for more information:

sklearn.ensemble.RandomForestClassifier<br/>
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

### Citation details:

Census Income Data Set<br/>
https://archive.ics.uci.edu/ml/datasets/census+income

### License:
Donation.

### Contact to send questions or comments about the model:

**Model creator:**<br/>
Juan Carlos Kuri Pinto<br/>
jckuri `@` gmail.com

**Dataset donor:**<br/>
Ronny Kohavi and Barry Becker<br/>
Data Mining and Visualization<br/>
Silicon Graphics.<br/>
e-mail: ronnyk `@` sgi.com for questions.

--------------------------------------------------------------------------------

## Intended Use

### Primary intended uses
This model helps to predict whether the salary of a person is greater than $50K
or not, based on many demographic features like age, workclass, education, 
marital status, occupation, race, sex, native country, and working hours per week.

### Primary intended users
This model is mostly educational. However, it could be used by marketing
strategists to inform them about who could have enough money to buy some product.
Credit cards could be also interested in this model.

### Out-of-scope use cases
This model is not accurate enough to be taken so seriously. This model could
inform users about who could have enough money. But their predictions cannot
be regarded as the last word.

--------------------------------------------------------------------------------

## Factors

### Relevant Factors
Relevant factors are demographic features like age, workclass, education, 
marital status, occupation, race, sex, native country, and working hours per week.

### Evaluation Factors
Evaluation factors are gender (male/female) and age group (young/old).

--------------------------------------------------------------------------------

## Metrics

This model was evaluated with the metrics precision, recall, Fbeta, and F1-score.

```
Precision = TruePositives / (TruePositives + FalsePositives)
```

```
Recall = TruePositives / (TruePositives + FalseNegatives)
```

A Gentle Introduction to the Fbeta-Measure for Machine Learning<br/>
https://machinelearningmastery.com/fbeta-measure-for-machine-learning/

```
Fbeta = ((1 + beta^2) * Precision * Recall) / (beta^2 * Precision + Recall)
```

Three common values for the beta parameter are as follows:

F0.5-Measure (beta=0.5): More weight on precision, less weight on recall.
F1-Measure (beta=1.0): Balance the weight on precision and recall.
F2-Measure (beta=2.0): Less weight on precision, more weight on recall

F-score<br/>
https://en.wikipedia.org/wiki/F-score

```
F1 = 2 * precision * recall / (precision + recall)
```

## Evaluation Data

Evaluation data is the test data split of this dataset:

Census Income Data Set<br/>
https://archive.ics.uci.edu/ml/datasets/census+income

**Preprocessing steps:**
- Spaces between commas were removed.
- Categorical features were transformed into one-hot encoding.

## Training Data

Training data is the training data split of this dataset:

Census Income Data Set<br/>
https://archive.ics.uci.edu/ml/datasets/census+income

**Preprocessing steps:**
- Spaces between commas were removed.
- Categorical features were transformed into one-hot encoding.

## Quantitative Analyses

Two variables were used for the quantitative analyses:
Age (young <=50 / old >50) and gender (male / female).
Four metrics were analyzed: Precision, recall, F-beta, and F1-score.
Unitary results were analyzed for the variables age and gender.
And intersectional results were analyzed for the same variables, creating 4
combinations: Young men, young women, old men, and old women.

```
SLICE			PRECISION	RECALL		F-BETA		F1-SCORE
Young Men		0.7875		0.6408		0.7066		0.7066
Young Women		0.7559		0.5026		0.6038		0.6038
Old Men			0.7848		0.6371		0.7033		0.7033
Old Women		0.6786		0.4524		0.5429		0.5429
Young			0.7831		0.6180		0.6908		0.6908
Old			0.7758		0.6184		0.6882		0.6882
Men			0.7868		0.6398		0.7057		0.7057
Women			0.7419		0.4936		0.5928		0.5928
Test Dataset		0.7812		0.6181		0.6901		0.6901
```

## Ethical Considerations

- Given this model is not 100% accurate, its predictions should not be used
  to discriminate people based on their demographic information. And model users
  should not use these model predictions to decide whether they will do
  businesses or not with the people studied.
- Salaries vary widely among countries. The salaries earned in developed 
  countries are higher than salaries earned in developing countries. So,
  the outcome of this model is not a good predictor for the socioeconomic status
  of citizens of a particular country.

## Caveats and Recommendations

- Given gender classes are binary (male/female), further work is needed to
  evaluate across a spectrum of genders.
- Salaries vary widely among countries. The salaries earned in developed 
  countries are higher than salaries earned in developing countries. High-class
  citizens of developing countries can earn less money than low-class citizens of
  developed countries.
