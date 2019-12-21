import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from helpers.bookmaker_selection import find_bookies_to_keep
from helpers.features import generate_bet_features, generate_match_features
from helpers.utils import data_prepare
from model import (rf_classification, rf_regression)

bookies_to_keep = find_bookies_to_keep('2018-01-01', '2019-12-01', 0.975)

bets_df = generate_bet_features(bookies_to_keep, na_ratio=0.15)
matches_df, test_match_ids = generate_match_features('2019-12-20', '2019-12-24')

final_df = data_prepare(bets_df, matches_df)

X_train = final_df.drop(["match_id", "over_under", "total_score", "result"], axis=1)
y_train = final_df[["result"]]

X_test = bets_df.merge(matches_df, on='match_id', how='outer')
X_test = X_test[X_test['match_id'].isin(test_match_ids)]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from helpers.utils import rps_metric
from sklearn.metrics import make_scorer

scaler = StandardScaler()
pca = PCA()
rf = RandomForestClassifier(random_state=42, oob_score = True)
pipe = Pipeline(steps=[('scaler', scaler), 
                       ('pca', pca), 
                       ('rf', rf)])
param_grid = { 
    'pca__n_components': [5, 8, 12],
    'rf__bootstrap': [True],
    'rf__n_estimators': [500],
    'rf__max_features': ['sqrt', 'log2', 3],
    'rf__min_samples_leaf' : [5]
}

kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=7)
scorer = make_scorer(rps_metric, 
                     greater_is_better=False, 
                     needs_proba=True)

search = GridSearchCV(pipe, param_grid, 
                      cv=kfold, scoring= scorer, n_jobs=-1)

search.fit(X_train, y_train)

X_test_clean = X_test.drop(['match_id'], axis=1).dropna()


output = search.predict_proba(X_test_clean)

output = pd.DataFrame(output, columns=search.classes_)
output['match_id'] = X_test.dropna().match_id.values
output = output[['match_id', 1, 0, 2]]

matches = pd.read_csv('data/matches.zip')
matches = matches[['match_id', 'match_hometeam_name', 'match_awayteam_name']]
matches = matches[matches['match_id'].isin(test_match_ids)]

output = output.merge(matches, on='match_id', how='outer')


from helpers.utils import create_output
create_output(output[['match_id', 1, 0, 2]])