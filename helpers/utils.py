import datetime as dt
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array

result_mapping = {1: [1, 0, 0],
                  0: [0, 1, 0],
                  2: [0, 0, 1]}

def ranked_probability_loss(obs, preds, change_order=True):
    """
    >>> y_true = [1, 1]
    >>> y_prob = [[0.5, 0.3, 0.2], [0.5, 0.2, 0.3]]
    >>> ranked_probability_loss(y_true, y_prob) # array([0.145, 0.17 ])

    >>> y_true = [1]
    >>> y_prob = [[0.7, 0.3, 0]]
    >>> ranked_probability_loss(y_true, y_prob) # array([0.045])
    """
    if change_order:
        preds = [np.concatenate([m[:2][::-1], m[2:]]) for m in preds]
    
    obs = check_array(obs, ensure_2d=False)
    preds = check_array(preds, ensure_2d=False)
    obs = np.array([result_mapping[i[0]] for i in obs])

    cum_diff = np.cumsum(preds, axis=1) - np.cumsum(obs, axis=1)
    result = np.sum(np.square(cum_diff), axis=1)/2
    return np.round(result, 5)

def rps_metric(obs, preds):
    return np.mean(ranked_probability_loss(obs, preds))

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
    year, week, day = timestamp.isocalendar()
    season = year - 1 if week < 27 else year
    is_weekend = day >= 5 or day == 1  
    return [timestamp, season, year, week, is_weekend]

def find_results(match_ids):
    matches = pd.read_csv('data/matches.zip')
    matches = matches[matches['match_id'].isin(match_ids)]
    matches['result'] = np.nan
    matches.loc[matches.match_hometeam_score > matches.match_awayteam_score, 
                'result'] = 1
    matches.loc[matches.match_hometeam_score == matches.match_awayteam_score, 
                'result'] = 0
    matches.loc[matches.match_hometeam_score < matches.match_awayteam_score, 
                'result'] = 2
    return matches[['match_id', 'result']]

def data_prepare(bets_df, matches_df):
    final_df = bets_df.merge(matches_df, on='match_id').dropna()
    matches = pd.read_csv("data/matches.zip")
    common_cols = ["match_id", "match_hometeam_score", "match_awayteam_score"]
    matches = matches[common_cols]

    matches["total_score"] = matches["match_hometeam_score"] + \
        matches["match_awayteam_score"]
        
    matches["over_under"] = np.nan
    matches.loc[matches.total_score >= 3, "over_under"] = "over"
    matches.loc[matches.total_score < 3, "over_under"] = "under"
    
    matches['result'] = np.nan
    matches.loc[matches.match_hometeam_score > matches.match_awayteam_score, 
                'result'] = 1
    matches.loc[matches.match_hometeam_score == matches.match_awayteam_score, 
                'result'] = 0
    matches.loc[matches.match_hometeam_score < matches.match_awayteam_score, 
                'result'] = 2
    matches.dropna(inplace=True)
    matches['result'] = matches['result'].astype(int)

    final_df = final_df.merge(matches, on="match_id")
    final_df = final_df.drop(["match_hometeam_score", "match_awayteam_score"], 
                             axis=1)
    return final_df.dropna()
    