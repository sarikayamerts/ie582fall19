import datetime as dt
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array

result_mapping = {1: [1, 0, 0],
                  0: [0, 1, 0],
                  2: [0, 0, 1]}

def ranked_probability_loss(obs, preds):
    """
    >>> y_true = [1, 1]
    >>> y_prob = [[0.5, 0.3, 0.2], [0.5, 0.2, 0.3]]
    >>> ranked_probability_loss(y_true, y_prob) # array([0.145, 0.17 ])

    >>> y_true = [1]
    >>> y_prob = [[0.7, 0.3, 0]]
    >>> ranked_probability_loss(y_true, y_prob) # array([0.045])
    """
    obs = check_array(obs, ensure_2d=False)
    preds = check_array(preds, ensure_2d=False)
    obs = np.array([result_mapping[i] for i in obs])

    cum_diff = np.cumsum(preds, axis=1) - np.cumsum(obs, axis=1)
    result = np.sum(np.square(cum_diff), axis=1)/2
    return np.round(result, 5)

def rps_metric(obs, preds):
    return 1 - ranked_probability_loss(obs, preds)

def create_output(df):
    """
    Converts dataframe to comma separated string
    """
    output_list = df.to_string(header=False,
                               index=False,
                               index_names=False).split('\n')
    output_string = ','.join([','.join(ele.split()) for ele in output_list])
    return output_string

def week_converter(timestamp):
    """
    year is 2019 for dates between 2019-07 and 2020-06, 
    22nd week just random splitter, 
    there might be better representation
    
    is_national is True for Friday, Saturday, Sunday, Monday 
    False otherwise
    """
    year, week, day = (timestamp - dt.timedelta(1)).isocalendar()
    year = year - 1 if week < 22 else year
    is_national = day >= 4
    return [year, week, is_national]

def find_results(match_ids):
    matches = pd.read_csv('data/matches.zip')
    matches = matches[matches['match_id'].isin(match_ids)]
    matches['result'] = np.where(matches.match_hometeam_score > matches.match_awayteam_score, 
                             1, 0)
    matches['result'] = np.where(matches.match_hometeam_score < matches.match_awayteam_score, 
                             2, matches.result)
    return matches[['match_id', 'result']]