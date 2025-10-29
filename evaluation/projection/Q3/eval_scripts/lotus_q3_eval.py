import pandas as pd
from rapidfuzz import process, fuzz
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

player_evi = pd.read_csv("datasets/rotowire/player_evidence_mine.csv")[['Player Name', 'nationality']].dropna(subset=['nationality']).head(args.size)

if args.provider == 'ollama' or args.provider == 'transformers':
    results_file = f"evaluation/projection/Q3/results/lotus_Q3_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider == 'vllm':
    results_file = f"evaluation/projection/Q3/results/lotus_Q3_map_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

lotus_evi = pd.read_csv(results_file)

df = player_evi.merge(lotus_evi, left_on='Player Name', right_on='Player Name', how='outer')
df['nationality_y'] = df['nationality_y'].str.replace('\n', '')

df["match"] = df.apply(
    lambda row: (
        isinstance(row["nationality_x"], str)
        and isinstance(row["nationality_y"], str)
        and len(row["nationality_y"]) <= 30
        and (
            row["nationality_y"].lower() in row["nationality_x"].lower()
            or row["nationality_x"].lower() in row["nationality_y"].lower()
            or fuzz.ratio(row["nationality_x"], row["nationality_y"]) >= 70
        )
    ),
    axis=1
)
print(f"Accuracy: {df['match'].mean():.2%}")