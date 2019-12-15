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

X = final_df.drop(["match_id", "over_under", "total_score"], axis=1)
y = final_df[["total_score"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.17, random_state=42)

CV_rfc = rf_classification(X_train, y_train)
CV_rfc = rf_regression(X_train, y_train)


# CV_rfc.best_params_
# CV_rfc.cv_results_

# accuracy_score(y_test, CV_rfc.best_estimator_.predict(X_test))


