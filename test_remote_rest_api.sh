echo "Testing GET:"
curl -X GET "$1"

#       age         workclass   fnlgt     education  education-num      marital-status         occupation   relationship   race     sex  capital-gain  capital-loss  hours-per-week native-country salary
#14160   27           Private  160178  Some-college             10            Divorced       Adm-clerical  Not-in-family  White  Female             0             0              38  United-States  <=50K   
#28868   29           Private  185908     Bachelors             13  Married-civ-spouse    Exec-managerial        Husband  Black    Male             0             0              55  United-States   >50K 

echo "\nResult of POST 1: "
curl -X POST "$1/predict_salary" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"age\":27, \"workclass\":\"Private\", \"fnlgt\":160178, \"education\":\"Some-college\", \"education_num\":10, \"marital_status\":\"Divorced\", \"occupation\":\"Adm-clerical\", \"relationship\":\"Not-in-family\", \"race\":\"White\", \"sex\":\"Female\", \"capital_gain\":0, \"capital_loss\":0, \"hours_per_week\":38, \"native_country\":\"United-States\"}"

echo "\nResult of POST 2: "
curl -X POST "$1/predict_salary" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"age\":29, \"workclass\":\"Private\", \"fnlgt\":185908, \"education\":\"Bachelors\", \"education_num\":13, \"marital_status\":\"Married-civ-spouse\", \"occupation\":\"Exec-managerial\", \"relationship\":\"Husband\", \"race\":\"Black\", \"sex\":\"Male\", \"capital_gain\":0, \"capital_loss\":0, \"hours_per_week\":55, \"native_country\":\"United-States\"}"

echo ""
