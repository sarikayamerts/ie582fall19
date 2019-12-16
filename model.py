from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import tree
from xgboost import XGBClassifier, XGBRegressor


def rf_classification(X, y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    rfc=RandomForestClassifier(random_state=42, oob_score = True)
    param_grid = { 
        'bootstrap': [True],
        'n_estimators': [500],
        'max_features': ['sqrt', 'log2', 15, 25, 40],
        'min_samples_leaf' : [5]
    }
    kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=7)

    CV_rfc = GridSearchCV(estimator=rfc, 
                        param_grid=param_grid, 
                        cv= kfold, 
                        verbose=True, n_jobs=-1, return_train_score=True)
    CV_rfc.fit(X, y)
    return CV_rfc

def rf_regression(X, y):
    rfc=RandomForestRegressor(random_state=42)
    param_grid = { 
        'bootstrap': [True],
        'n_estimators': [500],
        'max_features': ['sqrt', 'log2', 15, 25, 40],
        'min_samples_leaf' : [5],
        'max_depth': [4, 10]
    }

    CV_rfc = GridSearchCV(estimator=rfc, 
                        param_grid=param_grid, 
                        cv= 8, 
                        verbose=True, n_jobs=-1, return_train_score=True)
    CV_rfc.fit(X, y)
    return CV_rfc

def xgb_classifier(X,y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    xgb = XGBClassifier()
    kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=7)

    param_grid = {'objective':['binary:logistic'],
                  'learning_rate': [0.05, 0.1, 0.2],
                  'max_depth': [2, 4, 6],
                  'min_child_weight': [11],
                  'silent': [1],
                  'subsample': [0.7],
                  'colsample_bytree': [0.7],
                  'n_estimators': [100, 250, 500],
                  'seed': [1337]}
    
    CV_xgb = GridSearchCV(estimator=xgb, 
                        param_grid=param_grid, 
                        cv= kfold, 
                        verbose=True, n_jobs=-1, return_train_score=True)
    CV_xgb.fit(X, y)
    
    return CV_xgb

def xgb_regression(X,y):
    xgb = XGBRegressor()

    param_grid = {'objective':['reg:linear'],
                  'learning_rate': [0.05, 0.1, 0.2],
                  'max_depth': [2, 4, 6],
                  'min_child_weight': [11],
                  'silent': [1],
                  'subsample': [0.7],
                  'colsample_bytree': [0.7],
                  'n_estimators': [100, 250, 500],
                  'seed': [1337]}
    
    CV_xgb = GridSearchCV(estimator=xgb, 
                        param_grid=param_grid, 
                        cv= 8, 
                        verbose=True, n_jobs=-1, return_train_score=True)
    CV_xgb.fit(X, y)
    
    return CV_xgb

def dt_regression(X,y):
    
    dt = tree.DecisionTreeRegressor()

    param_grid = {'criterion': ["mse"],
              'min_samples_leaf':[5, 10, 15],
              'ccp_alpha': [0.0, 0.2, 0.6]}

    CV_dt = GridSearchCV(dt, param_grid, n_jobs=-1, 
                       verbose=2, refit=True)
    
    CV_dt.fit(X, y)

    return CV_dt

def dt_classifier(X,y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    dt = tree.DecisionTreeClassifier()

    kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=7)

    param_grid = {'criterion': ["gini"],
              'min_samples_leaf':[5, 10, 15],
              'ccp_alpha': [0.0, 0.2, 0.6]}

    CV_dt = GridSearchCV(dt, param_grid, n_jobs=-1,
                         cv = kfold,
                       verbose=2, refit=True)
    
    CV_dt.fit(X, y)

    return CV_dt
    


# import numpy as np
# import pandas as pd
# from data_prepare import prepare
# from bets_features import generate_bets_features
# from sklearn.ensemble import RandomForestClassifier
# from utils import rps_metric, ranked_probability_loss, find_results, create_output

# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import make_scorer

# class RandomForestClassifierRPS(RandomForestClassifier):
#     def score(self, X, y, sample_weight=None):
#         return rps_metric(y, self.predict(X))

# X, y, X_test, matches, test_matches, bets, final_bets = prepare(
#     '2019-12-14', '2019-12-16')
# bets_df = generate_bets_features()
# bets_df = bets_df.reset_index()
# test = bets_df[bets_df['match_id'].isin(test_matches.match_id)]
# X_train = bets_df[~bets_df['match_id'].isin(test_matches.match_id)]
# X_train = X_train[X_train['match_id'].isin(y.match_id)]
# y_train = y[y['match_id'].isin(X_train.match_id)]

# # results = find_results(X_test.match_id).sort_values('match_id')
# # X = X.merge(y, on='match_id')

# X_test = test.drop(['match_id'], axis=1)
# X_train = X_train.drop(['match_id'], axis=1)
# y_train = y_train.result.astype(int)
# rfc=RandomForestClassifierRPS(random_state=42, oob_score = True)
# param_grid = { 
#     'bootstrap': [True],
#     'n_estimators': [1000],
#     'max_features': ['auto'],
#     'max_depth' : [2,4,6,8,10]
# }
# kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=7)

# scorer = make_scorer(ranked_probability_loss, 
#                      greater_is_better=False, 
#                      needs_proba=True)
# CV_rfc = GridSearchCV(estimator=rfc, 
#                       param_grid=param_grid, 
#                       cv= kfold, 
#                       scoring= scorer, verbose=True, n_jobs=-1)
# CV_rfc.fit(X_train, y_train)


# ranked_probability_loss(
#     results.result,
#     # find_results(X.match_id).sort_values('match_id').result, 
#     CV_rfc.best_estimator_.predict_proba(X_test))

# out_df = pd.DataFrame(CV_rfc.best_estimator_.predict_proba(X_test))
# out_df = pd.concat([test.match_id.reset_index(drop=True), out_df], axis=1)
# out_df = test_matches[['match_id', 'match_hometeam_name', 'match_awayteam_name']].merge(out_df, on=['match_id'],how='left')
# create_output(out_df)
# clf.predict_proba(X_test)

# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import SelectFromModel


# sel_ = SelectFromModel(LogisticRegression(cv=10, normalize=True, tol=0.001, 
#                 selection='random', random_state=43))

# logistic_model = LogisticRegression(
#     multi_class='multinomial', 
#     solver='newton-cg', 
#     max_iter=1000)
# sel_.fit(X, y)
# sel_.get_support()



# model = XGBClassifier(needs_proba=True)

# label_encoded_y = LabelEncoder().fit_transform(y)

# scorer = make_scorer(ranked_probability_loss, 
#                      greater_is_better=False, 
#                      needs_proba=True)


# parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#               'objective':['multi:softprob'],
#               'learning_rate': [0.01, 0.05], #so called `eta` value
#               'max_depth': [2, 4],
#               'min_child_weight': [11],
#               'silent': [1],
#               'subsample': [0.6, 0.8],
#               'colsample_bytree': [0.7],
#               'n_estimators': [100, 250], #number of trees, change it to 1000 for better results
#               'seed': [1337]}


# clf = GridSearchCV(model, parameters, n_jobs=-1, 
#                    cv=kfold, 
#                    scoring= scorer,
#                    verbose=2, refit=True)

# clf.fit(X, label_encoded_y)

# clf.predict_proba(X_test.iloc[:,1:])
# results
# ranked_probability_loss(results.result, clf.predict_proba(X_test.iloc[:,1:]))