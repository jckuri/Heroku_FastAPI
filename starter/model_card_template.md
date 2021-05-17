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


**Features:**
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

**Label:**
salary: >50K, <=50K.

### Paper or other resource for more information:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

### Citation details:
https://archive.ics.uci.edu/ml/datasets/census+income

### License:
Donation.

### Contact to send questions or comments about the model:
jckuri@gmail.com

--------------------------------------------------------------------------------

## Intended Use

## Factors

## Metrics

## Evaluation Data

## Training Data

## Quantitative Analyses

## Ethical Considerations

## Caveats and Recommendations
