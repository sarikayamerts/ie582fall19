import datetime as dt
import pandas as pd

# bookies_to_keep = find_bookies_to_keep('2018-01-01', '2019-12-01', 0.95)

def find_bookies_to_keep(start_date, end_date, ratio):
    bets = pd.read_csv("data/bets.zip")
    matches = pd.read_csv("data/matches.zip")

    # Converting epoch column to datetime
    matches['timestamp'] = matches['epoch'].apply(
        lambda x: dt.datetime.fromtimestamp(x))
    bets['timestamp'] = bets['odd_epoch'].apply(
        lambda x: dt.datetime.fromtimestamp(x))

    matches = matches[(matches['timestamp'] > start_date) &
                      (matches['timestamp'] < end_date) &
                      (matches['league_id'] == 148)]

    matches = matches.dropna(
        subset=['match_status', 'match_hometeam_score',
                'match_awayteam_score'])

    match_ids = list(matches.match_id)
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

    bookies = final_bets.groupby('odd_bookmakers').count()[['match_id']].reset_index()
    bookies['total_matches'] = final_bets.match_id.nunique()
    bookies['ratio'] = bookies['match_id'] / bookies['total_matches']
    bookies.sort_values('ratio', ascending=False, inplace=True)
    bookies.reset_index(drop=True, inplace=True)
    bookies_to_keep = bookies[bookies['ratio'] > ratio]
    return list(bookies_to_keep.odd_bookmakers)