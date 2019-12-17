import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

training_set = pd.read_csv("training_data.csv")

def preprocess(data_set):
    data_set["DATE"] = pd.to_datetime((data_set.YEAR*10000+data_set.MONTH*100+data_set.DAY).apply(str),format='%Y%m%d')
    data_set["DAY_NUM"] =  (pd.to_datetime(data_set["DATE"]) - pd.to_datetime("2018-03-01"))
    one_hot = pd.get_dummies(data_set['REGION'],prefix="r_")
    data_set = data_set.drop('REGION',axis = 1)
    data_set = data_set.join(one_hot)
    one_hot2 = pd.get_dummies(data_set['TRX_TYPE'],prefix="d_")
    data_set = data_set.drop('TRX_TYPE',axis = 1)
    data_set = data_set.join(one_hot2)
    data_set["DAY_OF_WEEK"] = data_set["DATE"].dt.dayofweek
    data_set["DAY_NUM"] = data_set["DAY_NUM"].dt.days
    data_set = data_set.drop(['DATE'],axis=1)
    one_hot3 = pd.get_dummies(data_set['DAY_OF_WEEK'],prefix="w_")
    data_set = data_set.drop('DAY_OF_WEEK',axis = 1)
    data_set = data_set.join(one_hot3)
    return data_set

training_set = preprocess(training_set)
df = training_set.copy()

train_y = df.TRX_COUNT.values
train_x = df.drop(["TRX_COUNT"],axis=1)
train_x = train_x.drop(["IDENTITY"],axis=1).values

X_train, X_test, y_train, y_test = train_test_split(train_x,train_y,test_size=.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective':'reg:squarederror',
}

params['eval_metric'] = "mae"
num_boost_round = 1200

gridsearch_params = [
    (n_estimators, max_depth, min_child_weight)
    for n_estimators in (50, 100, 150, 200)
    for max_depth in range(9,12)
    for min_child_weight in range(5,8)
]

min_mae = float("Inf")
best_params = None
for n_estimators, max_depth, min_child_weight in gridsearch_params:
    params['n_estimators'] = n_estimators
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (n_estimators,max_depth,min_child_weight)
params['n_estimators'] = best_params[0]
params['max_depth'] = best_params[1]
params['min_child_weight'] = best_params[2]

gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(5,11)]
    for colsample in [i/10. for i in range(5,11)]
]

min_mae = float("Inf")
best_params = None
for subsample, colsample in reversed(gridsearch_params):
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
params['subsample'] = best_params[0]
params['colsample_bytree'] = best_params[1]

min_mae = float("Inf")
best_params = None
for eta in [.3, .2, .1, .05, .01, .005]:
    params['eta'] = eta
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=10)
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
params['eta'] = best_params

model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)

test_set = pd.read_csv("test_data.csv")
test_set = preprocess(test_set)
identities = test_set["IDENTITY"]
test_set = test_set.drop(['IDENTITY'],axis=1)
dtest = xgb.DMatrix(test_set.values)
predictions = model.predict(dtest)
results = pd.DataFrame(predictions)
results.to_csv('test_predictions.csv', index=False, header=False)
# Final version
