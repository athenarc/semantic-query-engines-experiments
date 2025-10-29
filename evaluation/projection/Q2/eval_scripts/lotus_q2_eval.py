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
    choices = df_labels.loc[df_labels['Game ID'] == game_id, 'Team Name'].tolist()
    group['matched_team'] = group['team_name'].apply(lambda x: match_name(x, choices))
    return group

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

df_labels = pd.read_csv('datasets/rotowire/team_labels.csv')
df_labels = df_labels[df_labels['Game ID'] < args.size]

# -------- Map --------
if args.provider == 'ollama':
    results_file = f"evaluation/projection/Q2/results/lotus_Q2_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider == 'vllm':
    results_file = f"evaluation/projection/Q2/results/lotus_Q2_map_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

lotus_labels = pd.read_csv(results_file)

lotus_labels = lotus_labels.groupby('Game ID', group_keys=False).apply(match_group)
lotus_labels.rename(columns={'wins': 'Wins', 'losses': 'Losses', 'total_points': 'Total points'}, inplace=True)

df = df_labels.merge(lotus_labels, left_on=['Game ID', 'Team Name'], right_on=['Game ID', 'matched_team'], how='left', indicator=True)

df.drop(columns=["Points in 4th quarter", "Percentage of field goals", "Rebounds", "Number of team assists", "Points in 3rd quarter", "Turnovers", "Percentage of 3 points", "Points in 1st quarter", "Points in 2nd quarter"], inplace=True)
cols = ["Wins", "Losses", "Total points"]

print("-------- Map --------")
for col in cols:
    xcol, ycol = f"{col}_x", f"{col}_y"
    df[f"{col}_match"] = (df[xcol].fillna(-1) == df[ycol].fillna(-1))

for col in cols:
    acc = df[f"{col}_match"].mean()
    print(f"{col} accuracy: {acc:.2%}")

total_accuracy = df[[f"{col}_match" for col in cols]].stack().mean()
print(f"Total accuracy: {total_accuracy:.2%}")

df_gtrue = df_labels[['Game ID', 'Team Name']]
df_pred = df[['Game ID', 'matched_team']].rename(columns={'matched_team': 'Team Name'})

merged = df_gtrue.merge(df_pred, on=['Game ID', 'Team Name'], how='outer', indicator=True)

TP = len(merged[merged['_merge'] == 'both'])
FP = len(merged[merged['_merge'] == 'right_only'])
FN = len(merged[merged['_merge'] == 'left_only'])

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"\nF1-score: {f1:.2f}\n")

if args.provider != 'vllm':
    exit(0)

# # -------- Extract --------
# if args.provider == 'ollama':
#     results_file = f"evaluation/projection/Q2/results/lotus_Q2_extract_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
# elif args.provider == 'vllm':
#     results_file = f"evaluation/projection/Q2/results/lotus_Q2_extract_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

# lotus_labels = pd.read_csv(results_file)

# lotus_labels = lotus_labels.groupby('Game ID', group_keys=False).apply(match_group)
# lotus_labels.rename(columns={'wins': 'Wins', 'losses': 'Losses', 'total_points': 'Total points'}, inplace=True)

# df = df_labels.merge(lotus_labels, left_on=['Game ID', 'Team Name'], right_on=['Game ID', 'matched_team'], how='left', indicator=True)

# df.drop(columns=["Points in 4th quarter", "Percentage of field goals", "Rebounds", "Number of team assists", "Points in 3rd quarter", "Turnovers", "Percentage of 3 points", "Points in 1st quarter", "Points in 2nd quarter"], inplace=True)
# cols = ["Wins", "Losses", "Total points"]

# print("-------- Extract --------")
# for col in cols:
#     xcol, ycol = f"{col}_x", f"{col}_y"
#     df[f"{col}_match"] = (df[xcol].fillna(-1) == df[ycol].fillna(-1))

# for col in cols:
#     acc = df[f"{col}_match"].mean()
#     print(f"{col} accuracy: {acc:.2%}")

# total_accuracy = df[[f"{col}_match" for col in cols]].stack().mean()
# print(f"Total accuracy: {total_accuracy:.2%}")

# df_gtrue = df_labels[['Game ID', 'Team Name']]
# df_pred = df[['Game ID', 'matched_team']].rename(columns={'matched_team': 'Team Name'})

# merged = df_gtrue.merge(df_pred, on=['Game ID', 'Team Name'], how='outer', indicator=True)

# TP = len(merged[merged['_merge'] == 'both'])
# FP = len(merged[merged['_merge'] == 'right_only'])
# FN = len(merged[merged['_merge'] == 'left_only'])

# precision = TP / (TP + FP) if (TP + FP) > 0 else 0
# recall = TP / (TP + FN) if (TP + FN) > 0 else 0
# f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"\nF1-score: {f1:.2f}\n")
