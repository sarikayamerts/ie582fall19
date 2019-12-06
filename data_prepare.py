import pandas as pd
import numpy as np
import datetime as dt
import logging
from utils import week_converter

logger = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
logger.setLevel(level)

def prepare(start_date, end_date):
    bets = pd.read_csv("data/bets.zip")
    matches = pd.read_csv("data/matches.zip")

    # Converting epoch column to datetime
    matches['timestamp'] = matches['epoch'].apply(
        lambda x: dt.datetime.fromtimestamp(x))
    bets['timestamp'] = bets['odd_epoch'].apply(
        lambda x: dt.datetime.fromtimestamp(x))

    matches[['year','week', 'is_weekend']] = pd.DataFrame(
        matches.timestamp.apply(week_converter).values.tolist(), 
        index=matches.index)

    start_date2 = dt.datetime.strptime(
        start_date, '%Y-%m-%d') - dt.timedelta(1)
    start_date2 = dt.datetime.strftime(start_date2, '%Y-%m-%d')
    end_date = dt.datetime.strptime(end_date, '%Y-%m-%d') + dt.timedelta(1)
    end_date = dt.datetime.strftime(end_date, '%Y-%m-%d')

    test_matches = matches[(matches['timestamp'] > start_date2) &
                        (matches['timestamp'] < end_date) &
                        (matches['league_id'] == 148)]
    test_matches = test_matches.sort_values('match_id')
    matches = matches[matches['timestamp'] < start_date]
    print('Number of test and train matches are {} and {}'
          .format(len(test_matches), len(matches)))
    matches = matches.dropna(
        subset=['match_status', 'match_hometeam_score', 
                'match_awayteam_score'])

    match_ids = list(test_matches.match_id.append(matches.match_id))
    bets = bets[bets['match_id'].isin(match_ids)]
    bets = bets[bets['value'] > 1]
    bets = bets[bets['variable'].isin(['odd_1', 'odd_x', 'odd_2'])]

    bets = bets.pivot_table(index=['match_id', 'odd_bookmakers', 'timestamp'],
                            columns='variable',
                            values='value').reset_index()
    bets = bets[['match_id', 'odd_bookmakers',
                'odd_1', 'odd_x', 'odd_2', 'timestamp']].dropna()

    final_bets = bets.groupby(['match_id', 'odd_bookmakers'],
                              as_index=False).last()
    
    for cols in ['odd_1', 'odd_x', 'odd_2']:
        final_bets['prob_'+cols] = 1 / final_bets[cols]

    final_bets['total'] = (final_bets['prob_odd_1'] + \
                          final_bets['prob_odd_x'] + \
                          final_bets['prob_odd_2'])

    for cols in ['odd_1', 'odd_x', 'odd_2']:
        final_bets['norm_prob_'+cols] = (final_bets['prob_'+cols] / 
                                         final_bets['total'])
    
    matches['result'] = np.where(
        matches.match_hometeam_score > matches.match_awayteam_score, 1, 0)
    matches['result'] = np.where(
        matches.match_hometeam_score < matches.match_awayteam_score, 
        2, matches.result)

    final_bets = final_bets.merge(matches[['match_id', 'result']], 
                              on='match_id', how='left')
    df = final_bets[
        ['match_id', 'odd_bookmakers', 'norm_prob_odd_1', 
         'norm_prob_odd_x', 'norm_prob_odd_2', 'result']]
    test = df[df.match_id.isin(test_matches.match_id)]
    train = df[df.match_id.isin(matches.match_id)]
    
    pivot_df = pd.pivot_table(train, 
               values=['norm_prob_odd_1', 'norm_prob_odd_x', 'norm_prob_odd_2'],
               columns=['odd_bookmakers'],
               index=['match_id', 'result'])
    pivot_df = pivot_df.reset_index()
    y = pivot_df.result
    X = pivot_df.drop(['match_id', 'result'], axis=1)
    
    pivot_df = pd.pivot_table(test, 
               values=['norm_prob_odd_1', 'norm_prob_odd_x', 'norm_prob_odd_2'],
               columns=['odd_bookmakers'],
               index=['match_id'])
    pivot_df = pivot_df.reset_index()
    X_test = pivot_df
    X = X[X_test.drop(['match_id'], axis=1).columns]
    print('Shape of X, y and X_test respectively is '
          .format(X.shape, y.shape, X_test.shape))
    X= X.fillna(1/3)
    X_test= X_test.fillna(1/3)
    
    return X, y, X_test, matches, test_matches, bets, final_bets