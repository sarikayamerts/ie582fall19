import numpy as np
import pandas as pd
from data_prepare import prepare
from sklearn.ensemble import RandomForestClassifier
from utils import rps_metric, ranked_probability_loss, find_results, create_output

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer

class RandomForestClassifierRPS(RandomForestClassifier):
    def score(self, X, y, sample_weight=None):
        return rps_metric(y, self.predict(X))

X, y, X_test, matches, test_matches, bets, final_bets = prepare(
    '2019-12-07', '2019-12-09')
# results = find_results(X_test.match_id).sort_values('match_id')
X = X.merge(y, on='match_id')

bookmaker_cols = [i for i in X.columns if i.startswith(
    ('Betclic', 'ComeOn', 'Tipsport.sk', 'STS.pl', 'bwin.fr', 'Pinnacle'))]

X = X[['match_id', *bookmaker_cols, 'result']].dropna().reset_index(drop=True)
X_test = X_test[['match_id', *bookmaker_cols]].dropna().reset_index(drop=True)
X_train = X.drop(['match_id', 'result'], axis=1)
y_train = X.result
rfc=RandomForestClassifierRPS(random_state=42, oob_score = True)
param_grid = { 
    'bootstrap': [True],
    'n_estimators': [1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [2,3,4]
}
kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=7)

scorer = make_scorer(ranked_probability_loss, 
                     greater_is_better=False, 
                     needs_proba=True)
CV_rfc = GridSearchCV(estimator=rfc, 
                      param_grid=param_grid, 
                      cv= kfold, 
                      scoring= scorer, verbose=True, n_jobs=-1)
CV_rfc.fit(X_train, y_train)


ranked_probability_loss(
    results.result,
    # find_results(X.match_id).sort_values('match_id').result, 
    CV_rfc.best_estimator_.predict_proba(X_test))

out_df = pd.DataFrame(CV_rfc.best_estimator_.predict_proba(X_test.iloc[:,1:]))
out_df = pd.concat([X_test.match_id, out_df], axis=1)
out_df = test_matches[['match_id', 'match_hometeam_name', 'match_awayteam_name']].merge(out_df, on=['match_id'],how='left')
create_output(out_df)
clf.predict_proba(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel


sel_ = SelectFromModel(LogisticRegression(cv=10, normalize=True, tol=0.001, 
                selection='random', random_state=43))

logistic_model = LogisticRegression(
    multi_class='multinomial', 
    solver='newton-cg', 
    max_iter=1000)
sel_.fit(X, y)
sel_.get_support()



model = XGBClassifier(needs_proba=True)

label_encoded_y = LabelEncoder().fit_transform(y)

scorer = make_scorer(ranked_probability_loss, 
                     greater_is_better=False, 
                     needs_proba=True)


parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['multi:softprob'],
              'learning_rate': [0.01, 0.05], #so called `eta` value
              'max_depth': [2, 4],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.6, 0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [100, 250], #number of trees, change it to 1000 for better results
              'seed': [1337]}


clf = GridSearchCV(model, parameters, n_jobs=-1, 
                   cv=kfold, 
                   scoring= scorer,
                   verbose=2, refit=True)

clf.fit(X, label_encoded_y)

clf.predict_proba(X_test.iloc[:,1:])
results
ranked_probability_loss(results.result, clf.predict_proba(X_test.iloc[:,1:]))