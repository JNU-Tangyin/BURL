from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import xgboost as xgb

# Spam
Spam_data = pd.read_csv('../Full Model/Spam_Embedding_by_FullModel.csv')
# News
News_data = pd.read_csv('../Full Model/News_Embedding_by_FullModel.csv')
# Malicious
Malicious_data = pd.read_csv('../Full Model/Malicious_Phish_Embedding_by_FullModel.csv')
# Classification
Classification_data = pd.read_csv('../Full Model/Classification_Embedding_by_FullModel.csv')
# App
App_data = pd.read_csv('../Full Model/App_Embedding_by_FullModel.csv')


dataset = [Spam_data, News_data, Malicious_data, Classification_data, App_data]

accuracies, precisions, f1s, recalls = [], [], [], []

for data in dataset:
    X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    labels_encoded, unique = pd.factorize(np.unique(Y))
    label_mapping = dict(zip(unique, labels_encoded))
    Y = np.vectorize(label_mapping.get)(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Parameters of Random Forest
    def RF_evaluate(n_estimators, max_depth, min_samples_split):
        params = {'n_estimators': int(n_estimators),
                  'max_depth': int(max_depth),
                  'min_samples_split': int(min_samples_split)}
        cv_result = cross_val_score(RandomForestClassifier(**params),
                                    X_train, Y_train,
                                    cv=5, scoring='accuracy')
        return np.mean(cv_result)

    # Bayesian Optimization
    RF_bo = BayesianOptimization(RF_evaluate, {'n_estimators': (10, 100),
                                               'max_depth': (3, 10),
                                               'min_samples_split': (2, 10)})
    RF_bo.maximize(init_points=3, n_iter=5)
    # Best Parameters
    params = RF_bo.max['params']
    optimal_params = {'n_estimators': int(params['n_estimators']),
                      'max_depth': int(params['max_depth']),
                      'min_samples_split': int(params['min_samples_split'])}
    RF_model = RandomForestClassifier(**optimal_params)
    RF_model.fit(X_train, Y_train)
    # Predict
    Y_pred = RF_model.predict(X_test)

    # Assessment
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='macro')
    recall = recall_score(Y_test,Y_pred, average='macro')
    f1 = f1_score(Y_test, Y_pred, average='macro')

    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}, Accuracy: {accuracy}')

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    accuracies.append(accuracy)

print(precisions)
print(recalls)
print(f1s)
print(accuracies)

print('----------------------------------------------------------------------------------------------')
accuracies, precisions, f1s, recalls = [], [], [], []

for data in dataset:
    X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    labels_encoded, unique = pd.factorize(np.unique(Y))
    label_mapping = dict(zip(unique, labels_encoded))
    Y = np.vectorize(label_mapping.get)(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Parameters of XGBoost
    def xgb_evaluate(max_depth, gamma, colsample_bytree):
        params = {'eval_metric': 'logloss',
                'max_depth': int(max_depth),
                'subsample': 0.8,
                'eta': 0.1,
                'gamma': gamma,
                'colsample_bytree': colsample_bytree}
        cv_result = cross_val_score(xgb.XGBClassifier(**params),
                                    X_train, Y_train,
                                    cv=5, scoring='accuracy')

        return np.mean(cv_result)

    # Bayesian Optimization
    xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 10),
                                              'gamma': (0, 1),
                                              'colsample_bytree': (0.3, 0.9)})

    xgb_bo.maximize(init_points=3, n_iter=5)

    # Best Parameters
    params = xgb_bo.max['params']
    print(params)
    optimal_params = {'max_depth': int(params['max_depth']),
                  'gamma': params['gamma'],
                  'colsample_bytree': params['colsample_bytree'],
                  'eval_metric': 'logloss'}
    model = xgb.XGBClassifier(**optimal_params)
    model.fit(X_train, Y_train)

    # Predict
    Y_pred = model.predict(X_test)

    # Assessment
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='macro')
    recall = recall_score(Y_test,Y_pred, average='macro')
    f1 = f1_score(Y_test, Y_pred, average='macro')

    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}, Accuracy: {accuracy}')

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    accuracies.append(accuracy)

print(precisions)
print(recalls)
print(f1s)
print(accuracies)