import pandas as pd
from rapidfuzz import process, fuzz
import argparse

def match_name(name, choices, scorer=fuzz.ratio, threshold=40):
    if not choices:
        return None
    match = process.extractOne(name, choices, scorer=scorer, score_cutoff=threshold)
    return match[0] if match else None

def match_group(group):
    game_id = group.name
    choices = player_labels.loc[player_labels['Game ID'] == game_id, 'Player Name'].tolist()
    group['matched_player'] = group['name'].apply(lambda x: match_name(x, choices))
    return group

def count_true_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['Points_gt'] == df['Points_pred']) & (df['Points_gt'] == 17.0)])

def count_false_positives(df):
    wrong_val_pred = len(df[df['Points_pred'] != 17]) 
    temp_df = df[df['Points_pred'] == 17]

    return wrong_val_pred + len(temp_df[(temp_df['_merge'] == 'both') & (temp_df['Points_gt'] != 17) & (temp_df['Points_pred'] == 17)])


def count_true_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['Points_gt'] != 17.0)])

def count_false_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['Points_gt'] == 17.0)])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
    args = parser.parse_args()

    player_labels = pd.read_csv('datasets/rotowire/player_labels.csv')
    player_labels = player_labels[player_labels['Game ID'] < args.size]
    player_labels = player_labels[['Game ID', 'Player Name', 'Points']]

    eleet_res = pd.read_csv(f'evaluation/selection/Q6/results/ELEET_Q6_{args.size}.csv')

    eleet_res = eleet_res.groupby('Game ID', group_keys=False).apply(match_group)

    eleet_res = eleet_res.drop(columns=['name', 'Game ID']).rename(columns={'matched_player' : 'Player Name'})

    df = player_labels.merge(eleet_res, on=['Player Name'], how='outer', suffixes=('_gt', '_pred'), indicator=True)

    df = df.dropna(subset=['Game ID']).drop(columns=['Game ID'])

    tp = count_true_positives(df)
    fp = count_false_positives(df)
    tn = count_true_negatives(df)
    fn = count_false_negatives(df)

    print(f"True Positives: {tp}"
    f"\nFalse Positives: {fp}"
    f"\nTrue Negatives: {tn}"
    f"\nFalse Negatives: {fn}")
    print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2f}")
