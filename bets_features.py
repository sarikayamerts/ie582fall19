import pandas as pd
import numpy as np
import datetime as dt
from utils import week_converter
from bookmaker_selection import find_bookies_to_keep
bookies_to_keep = find_bookies_to_keep('2018-01-01', '2019-12-01', 0.975)

bets = pd.read_csv("data/bets.zip")
bets = bets[bets["odd_bookmakers"].isin(bookies_to_keep)]

bets['timestamp'] = bets['odd_epoch'].apply(
    lambda x: dt.datetime.fromtimestamp(x))

bets = bets[bets['value'] > 1]
bets = bets[bets['variable'].isin(['odd_1', 'odd_x', 'odd_2'])]

bets = bets.pivot_table(index=['match_id', 'odd_bookmakers', 'timestamp'],
                        columns='variable',
                        values='value').reset_index()
bets = bets[['match_id', 'odd_bookmakers',
            'odd_1', 'odd_x', 'odd_2', 'timestamp']].dropna()

for cols in ['odd_1', 'odd_x', 'odd_2']:
    bets['prob_'+cols] = 1 / bets[cols]

bets['total'] = (bets['prob_odd_1'] + bets['prob_odd_x'] + bets['prob_odd_2'])

for cols in ['odd_1', 'odd_x', 'odd_2']:
    bets['norm_prob_'+cols] = (bets['prob_'+cols] / bets['total'])

bets = bets.sort_values(
    ['timestamp', 'match_id', 'odd_bookmakers']).reset_index(drop=True)

bets = bets[["match_id", "odd_bookmakers", "norm_prob_odd_1", 
             "norm_prob_odd_x", "norm_prob_odd_2"]]

bets.rename({"norm_prob_odd_1": "odd_1",
             "norm_prob_odd_x": "odd_x",
             "norm_prob_odd_2": "odd_2"}, axis=1, inplace=True)

bets_features = bets.groupby(['match_id', 'odd_bookmakers']).agg(
    {'odd_1': ['min', 'max', 'first', 'last', 'var', 'mean'],
     'odd_x': ['min', 'max', 'first', 'last', 'var', 'mean'],
     'odd_2': ['min', 'max', 'first', 'last', 'var', 'mean', 'size']})

bets_features.columns = bets_features.columns.map('{0[0]}_{0[1]}'.format)
bets_features.rename({"odd_2_size": "size"}, axis=1, inplace=True)
bets_features.fillna(0, inplace=True)
mean_bets_features = bets_features.groupby('match_id').mean()

bets_features_pivoted = bets_features.pivot_table(
    index=["match_id"],
    columns= ["odd_bookmakers"])

bets_features_pivoted.columns = bets_features_pivoted.columns.map('{0[1]}_{0[0]}'.format)

for col in bets_features.columns:
    selected_cols = [cols for cols in bets_features_pivoted.columns if col in cols]
    selected = bets_features_pivoted[selected_cols]
    bets_features_pivoted[selected_cols] = \
        bets_features_pivoted[selected_cols].fillna(value=selected.mean(axis=1), axis=0)

