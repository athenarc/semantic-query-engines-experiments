import pandas as pd
from rapidfuzz import process, fuzz
import argparse

def match_name(name, choices, scorer=fuzz.ratio, threshold=60):
    if not choices:
        return None
    match = process.extractOne(name, choices, scorer=scorer, score_cutoff=threshold)
    return match[0] if match else None

def match_group(group):
    game_id = group.name
    choices = df_labels.loc[df_labels['Game ID'] == game_id, 'Player Name'].tolist()
    group['matched_player'] = group['name'].apply(lambda x: match_name(x, choices))
    return group

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
args = parser.parse_args()

df_labels = pd.read_csv('datasets/rotowire/player_labels.csv')
df_labels = df_labels[df_labels['Game ID'] < args.size]

df_eleet = pd.read_csv(f'evaluation/projection/Q1/results/ELEET_Q1_{args.size}.csv')
df_eleet = df_eleet.groupby('Game ID', group_keys=False).apply(match_group)
df = df_labels.merge(df_eleet, left_on=['Game ID', 'Player Name'], right_on=['Game ID', 'matched_player'], how='left', indicator=True)
df_eleet.to_csv("df_eleet_matched.csv", index=False)

str_to_num = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "thirteen": 13.0,
    "fourteen": 14.0,
    "fifteen": 15.0,
    "sixteen": 16.0,
    "seventeen": 17.0,
    "eighteen": 18.0,
    "nineteen": 19.0,
    "twenty": 20.0,
}
cols_to_replace = ["Points_y", "Assists_y", "Total rebounds_y", "Blocks_y", "Steals_y"]

df[cols_to_replace ] = df[cols_to_replace].replace(str_to_num)
df[cols_to_replace] = df[cols_to_replace].apply(pd.to_numeric, errors='coerce')

df.drop(columns=["Defensive rebounds", "Offensive rebounds", "3-pointers attempted", "3-pointers made", "Field goals attempted", "Field goals made", "Free throws attempted", "Free throws made", "Minutes played", "Personal fouls", "Turnovers", "Field goal percentage", "Free throw percentage", "3-pointer percentage"], inplace=True)

cols = ["Points", "Assists", "Total rebounds", "Blocks", "Steals"]

for col in cols:
    xcol, ycol = f"{col}_x", f"{col}_y"
    df[f"{col}_match"] = (df[xcol].fillna(-1) == df[ycol].fillna(-1))

for col in cols:
    acc = df[f"{col}_match"].mean()
    print(f"{col} accuracy: {acc:.2%}")

total_accuracy = df[[f"{col}_match" for col in cols]].stack().mean()
print(f"Total accuracy: {total_accuracy:.2%}")

df_gtrue = df_labels[['Game ID', 'Player Name']]
df_pred = df_eleet[['Game ID', 'matched_player']].rename(columns={'matched_player': 'Player Name'})

merged = df_gtrue.merge(df_pred, on=['Game ID', 'Player Name'], how='outer', indicator=True)

TP = len(merged[merged['_merge'] == 'both'])
FP = len(merged[merged['_merge'] == 'right_only'])
FN = len(merged[merged['_merge'] == 'left_only'])

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")