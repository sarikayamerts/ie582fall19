import numpy as np
import pandas as pd
from data_prepare import prepare
from sklearn.ensemble import RandomForestClassifier
from utils import rps_metric, ranked_probability_loss, find_results

class RandomForestClassifierRPS(RandomForestClassifier):
    def score(self, X, y, sample_weight=None):
        return rps_metric(y, self.predict(X))

X, y, X_test, matches, test_matches, bets, final_bets = prepare('2019-11-29', '2019-12-02')
results = find_results(X_test.match_id)
X_test = X_test.drop(['match_id'], axis=1)

clf = RandomForestClassifierRPS(max_depth=10)
clf.fit(X, y)
clf.predict_proba(X_test)

output = ranked_probability_loss(results.result, clf.predict_proba(X_test))
output, np.mean(output)

