import pandas as pd
import numpy as np
import datetime as dt
from helpers.utils import week_converter
from itertools import chain

def generate_bet_features(bookies_to_keep):
    bets = pd.read_csv("data/bets.zip")
    bets = bets[bets["odd_bookmakers"].isin(bookies_to_keep)]

    bets['timestamp'] = bets['odd_epoch'].apply(
        lambda x: dt.datetime.fromtimestamp(x))

    bets = bets[bets['value'] > 1]
    bet_groups = [['odd_1', 'odd_x', 'odd_2'],
                    ['bts_yes', 'bts_no'],
                    ['o+1.5', 'u+1.5'],
                    ['o+2.5', 'u+2.5'],
                    ['o+3.5', 'u+3.5'],
                    ['o+4.5', 'u+4.5'],
                    ['o+5.5', 'u+5.5']]

    bets = bets[bets['variable'].isin(list(chain.from_iterable(bet_groups)))]

    bets = bets.pivot_table(index=['match_id', 'odd_bookmakers', 'timestamp'],
                            columns='variable',
                            values='value').reset_index()

    for bet_type in bet_groups:
        bets[bet_type] = bets[bet_type].rdiv(1)
        bets[bet_type] = bets[bet_type].div(bets[bet_type].sum(axis=1),
                                            axis=0)

    bets = bets.sort_values(
        ['timestamp', 'match_id', 'odd_bookmakers']).reset_index(drop=True)

    standart_bets = bet_groups.pop(0)
    new_bets = list(chain.from_iterable(bet_groups))
    # the reason i did in this way, we may want to use different stats for
    # odd1x2 types and over under types
    bets_features = bets.groupby(['match_id', 'odd_bookmakers']).agg({
        **{i: ['min', 'max', 'first', 'last', 'mean', 'var', 'size'] 
            for i in standart_bets},
        **{i: ['min', 'max', 'first', 'last', 'mean', 'var', 'size'] 
            for i in new_bets}})

    bets_features.columns = bets_features.columns.map('{0[0]}_{0[1]}'.format)
    mean_bets_features = bets_features.groupby('match_id').mean()

    bets_features_pivoted = bets_features.pivot_table(
        index=["match_id"],
        columns= ["odd_bookmakers"])

    bets_features_pivoted.columns = bets_features_pivoted.columns.map('{0[1]}_{0[0]}'.format)

    for cols in bets_features_pivoted:
        mean_col = '_'.join(cols.split('_')[1:])
        bets_features_pivoted[cols] = bets_features_pivoted[cols].combine_first(
            mean_bets_features[mean_col])

    return bets_features_pivoted

def generate_match_features():
    matches = pd.read_csv("data/matches.zip")
    matches['timestamp'] = matches['epoch'].apply(
        lambda x: dt.datetime.fromtimestamp(x))
    matches[['date', 'season', 'year', 'week', 'is_weekend']] = \
        pd.DataFrame(matches.timestamp.apply(week_converter).values.tolist(), 
                        index=matches.index)
    matches = matches.sort_values("date")

    away_side = matches[["match_awayteam_id", "match_awayteam_name", 
                            "match_id", "season", "date", 
                            "match_awayteam_score", "match_hometeam_score"]]
    home_side = matches[["match_hometeam_id", "match_hometeam_name", 
                            "match_id", "season", "date", 
                            "match_hometeam_score", "match_awayteam_score"]]
    away_side["HomeAway"] = "Away"
    home_side["HomeAway"] = "Home"
    away_side.columns = ['team_id', 'team_name', "match_id", 
                            "season", "date", "scored", "conceded", "home_away"]
    home_side.columns = ['team_id', 'team_name', "match_id", 
                            "season", "date", "scored", "conceded", "home_away"]
    team_match = pd.concat([away_side, home_side])
    team_match = team_match.sort_values("date").reset_index(drop=True)

    team_match['point'] = np.nan
    team_match.loc[team_match['scored'] > team_match['conceded'], 'point'] = 3
    team_match.loc[team_match['scored'] == team_match['conceded'], 'point'] = 1
    team_match.loc[team_match['scored'] < team_match['conceded'], 'point'] = 0

    team_match['won'] = 0
    team_match.loc[team_match['point'] == 3, 'won'] = 1
    team_match['draw'] = 0
    team_match.loc[team_match['point'] == 1, 'draw'] = 1
    team_match['lost'] = 0
    team_match.loc[team_match['point'] == 0, 'lost'] = 1
    team_match['clean_sheet'] = 0
    team_match.loc[team_match['conceded'] == 0, 'clean_sheet'] = 1
    team_match['has_scored'] = 0
    team_match.loc[team_match['scored'] > 0, 'has_scored'] = 0
    team_match['over25'] = 0
    team_match.loc[team_match['scored'] + team_match['conceded'] > 2, 'over25'] = 1
    team_match['under25'] = 0
    team_match.loc[team_match['scored'] + team_match['conceded'] < 3, 'under25'] = 1

    roll1 = lambda x: x.rolling(1).mean().shift()
    roll5 = lambda x: x.rolling(5, min_periods = 1).mean().shift()
    historic = lambda x: x.expanding().mean().shift()

    team_match["point1"] = team_match.groupby(
        ["season", "team_id"]).point.apply(roll1).reset_index(0,drop=True)
    team_match["goal_scored1"] = team_match.groupby(
        ["season", "team_id"]).scored.apply(roll1).reset_index(0,drop=True)
    team_match["goal_conceded1"] = team_match.groupby(
        ["season", "team_id"]).conceded.apply(roll1).reset_index(0,drop=True)
    team_match["total_goals1"] = team_match["goal_conceded1"] + team_match["goal_scored1"]

    team_match["point5"] = team_match.groupby(
        ["season", "team_id"]).point.apply(roll5).reset_index(0,drop=True)
    team_match["goal_scored5"] = team_match.groupby(
        ["season", "team_id"]).scored.apply(roll5).reset_index(0,drop=True)
    team_match["goal_conceded5"] = team_match.groupby(
        ["season", "team_id"]).conceded.apply(roll5).reset_index(0,drop=True)
    team_match["clean_sheet5"] = team_match.groupby(
        ["season", "team_id"]).clean_sheet.apply(roll5).reset_index(0,drop=True)
    team_match["over25_ratio5"] = team_match.groupby(
        ["season", "team_id"]).over25.apply(roll5).reset_index(0,drop=True)
    team_match["under25_ratio5"] = team_match.groupby(
        ["season", "team_id"]).under25.apply(roll5).reset_index(0,drop=True)
    team_match["total_goals5"] = team_match["goal_conceded5"] + team_match["goal_scored5"]

    team_match["point1_pos"] = team_match.groupby(
        ["season", "team_id", "home_away"]).point.apply(roll1).reset_index(0,drop=True)
    team_match["goal_scored1_pos"] = team_match.groupby(
        ["season", "team_id", "home_away"]).scored.apply(roll1).reset_index(0,drop=True)
    team_match["goal_conceded1_pos"] = team_match.groupby(
        ["season", "team_id", "home_away"]).conceded.apply(roll1).reset_index(0,drop=True)

    team_match["performance_season"] = team_match.groupby(
        ["season", "team_id"]).point.apply(historic).reset_index(0,drop=True)

    team_match["draw_ratio"] = team_match.groupby(
        ["team_id"]).draw.apply(historic).reset_index(0,drop=True)
    team_match["win_ratio"] = team_match.groupby(
        ["team_id"]).won.apply(historic).reset_index(0,drop=True)
    team_match["lost_ratio"] = team_match.groupby(
        ["team_id"]).lost.apply(historic).reset_index(0,drop=True)
    team_match["over25_ratio"] = team_match.groupby(
        ["season", "team_id"]).over25.apply(historic).reset_index(0,drop=True)
    team_match["under25_ratio"] = team_match.groupby(
        ["season", "team_id"]).under25.apply(historic).reset_index(0,drop=True)

    team_match["draw_ratio_season"] = team_match.groupby(
        ["season", "team_id"]).draw.apply(historic).reset_index(0,drop=True)
    team_match["win_ratio_season"] = team_match.groupby(
        ["season", "team_id"]).won.apply(historic).reset_index(0,drop=True)
    team_match["lost_ratio_season"] = team_match.groupby(
        ["season", "team_id"]).lost.apply(historic).reset_index(0,drop=True)
    team_match["over25_ratio_season"] = team_match.groupby(
        ["season", "team_id"]).over25.apply(historic).reset_index(0,drop=True)
    team_match["under25_ratio_season"] = team_match.groupby(
        ["season", "team_id"]).under25.apply(historic).reset_index(0,drop=True)

    team_match["draw_ratio_pos"] = team_match.groupby(
        ["team_id", "home_away"]).draw.apply(historic).reset_index(0,drop=True)
    team_match["win_ratio_pos"] = team_match.groupby(
        ["team_id", "home_away"]).won.apply(historic).reset_index(0,drop=True)
    team_match["lost_ratio_pos"] = team_match.groupby(
        ["team_id", "home_away"]).lost.apply(historic).reset_index(0,drop=True)

    team_match["draw_ratio_season_pos"] = team_match.groupby(
        ["season", "team_id", "home_away"]).draw.apply(historic).reset_index(0,drop=True)
    team_match["win_ratio_season_pos"] = team_match.groupby(
        ["season", "team_id", "home_away"]).won.apply(historic).reset_index(0,drop=True)
    team_match["lost_ratio_season_pos"] = team_match.groupby(
        ["season", "team_id", "home_away"]).lost.apply(historic).reset_index(0,drop=True)
    team_match["over25_ratio_season_pos"] = team_match.groupby(
        ["season", "team_id"]).over25.apply(historic).reset_index(0,drop=True)
    team_match["under25_ratio_season_pos"] = team_match.groupby(
        ["season", "team_id"]).under25.apply(historic).reset_index(0,drop=True)

    match_id_pos = team_match.columns.get_loc("match_id")
    point1_pos = team_match.columns.get_loc("point1")
    len_cols = len(team_match.columns)
    
    cols = list(range(match_id_pos,match_id_pos+1)) + list(range(point1_pos, len_cols))
    home = team_match[team_match["home_away"] == 'Home'].iloc[:, cols]
    away = team_match[team_match["home_away"] == 'Away'].iloc[:, cols]
    team_stats = home.merge(away, on='match_id', how='inner', suffixes=('_home', '_away'))
    
    team_stats["point5_diff"] = team_stats["point5_home"] - team_stats["point5_away"]
    team_stats["point1_diff"] = team_stats["point1_home"] - team_stats["point1_away"]

    team_stats["performance_season_diff"] = team_stats["performance_season_home"] - team_stats["performance_season_away"]
    team_stats["exp_goal5"] = (team_stats["total_goals5_home"] + team_stats["total_goals5_away"])/2
    team_stats["exp_goal1"] = (team_stats["total_goals1_home"] + team_stats["total_goals1_away"])/2
    
    team_stats = team_stats[matches_df["point5_home"].notna()]

    return team_stats