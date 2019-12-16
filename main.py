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
matches_df = generate_match_features()

final_df = data_prepare(bets_df, matches_df)

final_df = data_prepare(bets_df, matches_df)

matches = pd.read_csv("data/matches.zip")
matches['timestamp'] = matches['epoch'].apply(
    lambda x: dt.datetime.fromtimestamp(x))

final_df = final_df.merge(matches[["match_id", "timestamp"]], on = "match_id")

final_df = final_df.sort_values("timestamp")
all_length = len(final_df)
test_size = int(np.round(len(final_df)/5))

final_df["test"] = 0
final_df.loc[final_df.tail(test_size).index, 'test'] = 1

X = final_df.drop(["match_id", "over_under", "total_score"], axis=1)
y = final_df[["total_score", "test"]]
X = final_df.drop(["match_id", "over_under", "total_score"], axis=1)
y = final_df[["total_score"]]

X_train = X[X["test"] == 0].drop(["test", "timestamp"], axis = 1)
X_test = X[X["test"] == 1].drop(["test", "timestamp"], axis = 1)
y_train_reg = y[y["test"] == 0]["total_score"]
y_test_reg = y[y["test"] == 1]["total_score"]
y_train_class = y_train_reg > 2.5
y_test_class = y_test_reg > 2.5

CV_rfc = rf_classification(X_train, y_train)
CV_rfc = rf_regression(X_train, y_train)


# CV_rfc.best_params_
# CV_rfc.cv_results_

# accuracy_score(y_test, CV_rfc.best_estimator_.predict(X_test))


