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
    away_side.columns = ['TeamId', 'TeamName', "MatchId", 
                            "Season", "Date", "Scored", "Conceded", "HomeAway"]
    home_side.columns = ['TeamId', 'TeamName', "MatchId", 
                            "Season", "Date", "Scored", "Conceded", "HomeAway"]
    team_match = pd.concat([away_side, home_side])
    team_match = team_match.sort_values("Date").reset_index(drop=True)

    team_match['Point'] = np.nan
    team_match.loc[team_match['Scored'] > team_match['Conceded'], 'Point'] = 3
    team_match.loc[team_match['Scored'] == team_match['Conceded'], 'Point'] = 1
    team_match.loc[team_match['Scored'] < team_match['Conceded'], 'Point'] = 0

    team_match['Won'] = 0
    team_match.loc[team_match['Point'] == 3, 'Won'] = 1
    team_match['Draw'] = 0
    team_match.loc[team_match['Point'] == 1, 'Draw'] = 1
    team_match['Lost'] = 0
    team_match.loc[team_match['Point'] == 0, 'Lost'] = 1

    team_match["SeasonOrder"] = team_match.groupby(
        ["TeamId", "Season"])["Date"].rank("dense", ascending=True)
    team_match["OverallOrder"] = team_match.groupby(
        ["TeamId"])["Date"].rank("dense", ascending=True)

    roll1 = lambda x: x.rolling(1).sum().shift()
    roll5 = lambda x: x.rolling(5).sum().shift()
    historic = lambda x: x.expanding().mean().shift()

    team_match["Point1"] = team_match.groupby(
        ["Season", "TeamId"]).Point.apply(roll1).reset_index(0,drop=True)
    team_match["GoalScored1"] = team_match.groupby(
        ["Season", "TeamId"]).Scored.apply(roll1).reset_index(0,drop=True)
    team_match["GoalConceded1"] = team_match.groupby(
        ["Season", "TeamId"]).Conceded.apply(roll1).reset_index(0,drop=True)

    team_match["Point5"] = team_match.groupby(
        ["Season", "TeamId"]).Point.apply(roll5).reset_index(0,drop=True)
    team_match["GoalScored5"] = team_match.groupby(
        ["Season", "TeamId"]).Scored.apply(roll5).reset_index(0,drop=True)
    team_match["GoalConceded5"] = team_match.groupby(
        ["Season", "TeamId"]).Conceded.apply(roll5).reset_index(0,drop=True)

    team_match["Point1Pos"] = team_match.groupby(
        ["Season", "TeamId", "HomeAway"]).Point.apply(roll1).reset_index(0,drop=True)
    team_match["GoalScored1Pos"] = team_match.groupby(
        ["Season", "TeamId", "HomeAway"]).Scored.apply(roll1).reset_index(0,drop=True)
    team_match["GoalConceded1Pos"] = team_match.groupby(
        ["Season", "TeamId", "HomeAway"]).Conceded.apply(roll1).reset_index(0,drop=True)

    team_match["PerformanceSeason"] = team_match.groupby(
        ["Season", "TeamId"]).Point.apply(historic).reset_index(0,drop=True)

    team_match["DrawRatio"] = team_match.groupby(
        ["TeamId"]).Draw.apply(historic).reset_index(0,drop=True)
    team_match["WinRatio"] = team_match.groupby(
        ["TeamId"]).Won.apply(historic).reset_index(0,drop=True)
    team_match["LostRatio"] = team_match.groupby(
        ["TeamId"]).Lost.apply(historic).reset_index(0,drop=True)

    team_match["DrawRatioSeason"] = team_match.groupby(
        ["Season", "TeamId"]).Draw.apply(historic).reset_index(0,drop=True)
    team_match["WinRatioSeason"] = team_match.groupby(
        ["Season", "TeamId"]).Won.apply(historic).reset_index(0,drop=True)
    team_match["LostRatioSeason"] = team_match.groupby(
        ["Season", "TeamId"]).Lost.apply(historic).reset_index(0,drop=True)

    team_match["DrawRatioPos"] = team_match.groupby(
        ["TeamId", "HomeAway"]).Draw.apply(historic).reset_index(0,drop=True)
    team_match["WinRatioPos"] = team_match.groupby(
        ["TeamId", "HomeAway"]).Won.apply(historic).reset_index(0,drop=True)
    team_match["LostRatioPos"] = team_match.groupby(
        ["TeamId", "HomeAway"]).Lost.apply(historic).reset_index(0,drop=True)

    team_match["DrawRatioSeasonPos"] = team_match.groupby(
        ["Season", "TeamId", "HomeAway"]).Draw.apply(historic).reset_index(0,drop=True)
    team_match["WinRatioSeasonPos"] = team_match.groupby(
        ["Season", "TeamId", "HomeAway"]).Won.apply(historic).reset_index(0,drop=True)
    team_match["LostRatioSeasonPos"] = team_match.groupby(
        ["Season", "TeamId", "HomeAway"]).Lost.apply(historic).reset_index(0,drop=True)

    team_match = team_match.drop(["Draw", "Won", "Lost"], axis = 1)
    cols = list(range(2,3)) + list(range(11, 33))
    home = team_match[team_match["HomeAway"] == 'Home'].iloc[:, cols]
    away = team_match[team_match["HomeAway"] == 'Away'].iloc[:, cols]
    team_stats = home.merge(away, on='MatchId', how='inner', suffixes=('_Home', '_Away'))
    return team_stats