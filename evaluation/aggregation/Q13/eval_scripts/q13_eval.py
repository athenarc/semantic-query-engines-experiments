import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=728, const=728, type=int, help="The input size")
args = parser.parse_args()

player_labels = pd.read_csv("datasets/rotowire/player_labels.csv")
player_labels = player_labels[player_labels['Game ID'] < args.size]

stats = ["Points", "Assists", "Total rebounds", "Steals", "Blocks"]

def is_triple_double(row):
    count = sum(row[stat] >= 10 for stat in stats if stat in row)
    return count >= 3


player_labels["Triple Double"] = player_labels.apply(is_triple_double, axis=1)

triple_double_counts = player_labels[player_labels["Triple Double"]] \
    .groupby("Player Name") \
    .size() \
    .reset_index(name="Triple Double Count")

best_player = triple_double_counts.sort_values("Triple Double Count", ascending=False).iloc[0]

print("Most triple doubles:", best_player["Player Name"])
print("Count:", best_player["Triple Double Count"])
