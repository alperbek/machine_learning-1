
#from google.colab import drive
#drive.mount('/content/gdrive')

import pandas as pd
import math
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

Xtrain = pd.read_csv("gdrive/My Drive/hw8/hw08_training_data.csv").drop(columns = ['ID'])
Xtrain = pd.get_dummies(Xtrain)
ytrain = pd.read_csv("gdrive/My Drive/hw8/hw08_training_label.csv").drop(columns = ['ID'])

test = pd.read_csv("gdrive/My Drive/hw8/hw08_test_data.csv")
test_ids = test['ID']
results = pd.DataFrame(test_ids)

test = pd.get_dummies(test.drop(columns = ['ID']))

train_dif_test = [i for i in Xtrain.columns.tolist() if i not in test.columns.tolist()]
test_dif_train = [i for i in test.columns.tolist() if i not in Xtrain.columns.tolist()]

Xtrain = Xtrain.assign(**dict.fromkeys(test_dif_train, 0))
test = test.assign(**dict.fromkeys(train_dif_test, 0))

Xtrain, Xeval, ytrain, yeval = train_test_split(Xtrain, ytrain,  test_size=0.05, random_state=1)

yeval = yeval.rename(columns={"TARGET_1": 1,"TARGET_2": 2,"TARGET_3": 3,"TARGET_4": 4,"TARGET_5": 5,"TARGET_6": 6})

ytrain = ytrain.rename(columns={"TARGET_1": 1,"TARGET_2": 2,"TARGET_3": 3,"TARGET_4": 4,"TARGET_5": 5,"TARGET_6": 6})

for i in range(1,7):

  valid_evals = pd.isna(yeval[i]) == False
  X_eval = Xeval[valid_evals]
  X_eval = X_eval.values
  y_eval = yeval[valid_evals]
  y_eval = y_eval[i].values
  deval = xgb.DMatrix(X_eval, label = y_eval)


  valid_indexes = pd.isna(ytrain[i]) == False

  x_train = Xtrain[valid_indexes].values
  y_train = ytrain[valid_indexes]

  target_labels = y_train[i].values
  dtrain = xgb.DMatrix(x_train, label = target_labels)
  dtest = xgb.DMatrix(test.values)

  params = {}
  params["objective"] =  "binary:logistic"
  params["booster"] = "gbtree"
  params["max_depth"] = 7
  params["eval_metric"] = 'auc'
  params["subsample"] = 0.8
  params["gamma"] = 1
  params["colsample_bytree"] = 0.8
  params["silent"] = 1
  params["seed"] = 4
  params["eta"] = 0.1
  params["alpha"] = 0.5
  params["lambda"] = 1.5

  num_boost_round = 999

  gridsearch_params = [
    (n_estimators, max_depth, min_child_weight)
    for n_estimators in (100, 150)
    for max_depth in range(9,11)
    for min_child_weight in range(5,7)
  ]

  max_auc = -1 * float("Inf")
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
          metrics = "auc",
          early_stopping_rounds=10
      )
      mean_auc = cv_results['test-auc-mean'].max()
      boost_rounds = cv_results['test-auc-mean'].idxmax()
      if mean_auc > max_auc:
          max_auc = mean_auc
          best_params = (n_estimators,max_depth,min_child_weight)
  params['n_estimators'] = best_params[0]
  params['max_depth'] = best_params[1]
  params['min_child_weight'] = best_params[2]

  gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in  (0.7, 1, 0.05)]
    for colsample in [i/10. for i in range(5,11)]
  ]
  max_auc = -1 * float("Inf")
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
          metrics = "auc",
          early_stopping_rounds=10
      )
      mean_auc = cv_results['test-auc-mean'].max()
      boost_rounds = cv_results['test-auc-mean'].idxmax()
      if mean_auc > max_auc:
          max_auc = mean_auc
          best_params = (subsample,colsample)
  params['subsample'] = best_params[0]
  params['colsample_bytree'] = best_params[1]

  max_auc = -1 * float("Inf")
  best_params = None
  for eta in [ 0.025, 0.25, 0.025]:
      params['eta'] = eta
      cv_results = xgb.cv(
              params,
              dtrain,
              num_boost_round=num_boost_round,
              seed=42,
              nfold=5,
              metrics = "auc",
              early_stopping_rounds=10)
      mean_auc = cv_results['test-auc-mean'].max()
      boost_rounds = cv_results['test-auc-mean'].idxmax()
      if mean_auc > max_auc:
          max_auc = mean_auc
          best_params = eta
  params['eta'] = best_params

  model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtrain, "train")],
    early_stopping_rounds=10
  )

  eval_predictions = model.predict(deval)
  eval_results = pd.DataFrame(eval_predictions)

  score = roc_auc_score(y_eval, eval_predictions)
  print("Eval score for "+ str(i) + ":" + str(score))

  predictions = model.predict(dtest)
  results[i] =  predictions

results = results.rename(columns={ 1: "TARGET_1", 2:"TARGET_2", 3: "TARGET_3" ,4:"TARGET_4" , 5: "TARGET_5" , 6: "TARGET_6"  })
results.to_csv("hw08_test_predictions.csv", index=False)
# Final version
