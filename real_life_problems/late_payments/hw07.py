#from google.colab import drive
#drive.mount('/content/gdrive')

import pandas as pd
import math
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

for i in range(1,4):

  Xtrain = pd.read_csv("gdrive/My Drive/hw7/hw07_target{}_training_data.csv".format(i)).drop(columns = ['ID'])
  Xtrain = pd.get_dummies(Xtrain)
  ytrain = pd.read_csv("gdrive/My Drive/hw7/hw07_target{}_training_label.csv".format(i)).drop(columns = ['ID']).values

  Xtrain = Xtrain.fillna(Xtrain.mean())
  test = pd.read_csv("gdrive/My Drive/hw7/hw07_target{}_test_data.csv".format(i))
  test_ids = test['ID']
  test = pd.get_dummies(test.drop(columns = ['ID']))

  train_dif_test = [i for i in Xtrain.columns.tolist() if i not in test.columns.tolist()]
  test_dif_train = [i for i in test.columns.tolist() if i not in Xtrain.columns.tolist()]

  Xtrain = Xtrain.assign(**dict.fromkeys(test_dif_train, 0))
  test = test.assign(**dict.fromkeys(train_dif_test, 0))

  Xtrain, Xeval, ytrain, yeval = train_test_split(Xtrain, ytrain,  test_size=0.05, random_state=1)

  dtrain = xgb.DMatrix(Xtrain.values, label = ytrain)
  deval = xgb.DMatrix(Xeval.values, label = yeval)
  dtest = xgb.DMatrix(test.values)

  params = {}
  params["objective"] =  "binary:logistic"
  params["booster"] = "gbtree"
  params["max_depth"] = 7
  params["eval_metric"] = 'auc'
  params["subsample"] = 0.8
  params["colsample_bytree"] = 0.8
  params["silent"] = 1
  params["seed"] = 4
  params["eta"] = 0.1

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
      for subsample in [i/10. for i in range(5,11)]
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
  for eta in [.3, .2, .1, .05, .01, .005]:
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
  print("Eval roc_auc_score" + str(i) + " :" + str(roc_auc_score(yeval, eval_predictions)))

  predictions = model.predict(dtest)
  results = pd.DataFrame(test_ids)
  results['TARGET'] =  predictions
  results.to_csv("hw07_target{}_test_predictions.csv".format(i),index=False)
  # Final version 2
