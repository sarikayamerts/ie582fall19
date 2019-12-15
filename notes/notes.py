from helpers.bookmaker_selection import find_bookies_to_keep
from helpers.features import generate_bet_features

bookies_to_keep = find_bookies_to_keep('2018-01-01', '2019-12-01', 0.975)

bets_df = generate_bet_features(bookies_to_keep)


import pandas as pd

pdf = pd.DataFrame(bets_df.columns)
pdf.columns = ['index']
pdf['bet_type'] = pdf['index'].apply(lambda x: '_'.join(x.split('_')[1:-1]))
pdf['stats'] = pdf['index'].apply(lambda x: '_'.join([x.split('_')[-1]]))
pdf['bookie'] = pdf['index'].apply(lambda x: '_'.join([x.split('_')[0]]))
pdf.groupby(['bet_type']).count()[['index']].sort_values('index', ascending=False)
pdf.groupby(['stats']).count()[['index']].sort_values('index', ascending=False)
pdf.groupby(['bookie']).count()[['index']].sort_values('index', ascending=False)
