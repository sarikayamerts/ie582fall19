from bookmaker_selection import find_bookies_to_keep
from features import generate_bet_features, generate_match_features

bookies_to_keep = find_bookies_to_keep('2018-01-01', '2019-12-01', 0.975)

bets_df = generate_bet_features(bookies_to_keep)
matches_df = generate_match_features()
