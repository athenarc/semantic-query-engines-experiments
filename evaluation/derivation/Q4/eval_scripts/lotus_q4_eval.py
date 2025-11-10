import pandas as pd
from rapidfuzz import process, fuzz
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

player_evi = pd.read_csv("datasets/rotowire/player_evidence_mine.csv")[['Player Name', 'birth_date']].head(args.size)

if args.provider == 'ollama':
    results_file = f"evaluation/derivation/Q4/results/lotus_Q4_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider == 'vllm':
    results_file = f"evaluation/derivation/Q4/results/lotus_Q4_map_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

lotus_evi = pd.read_csv(results_file)
lotus_evi.rename(columns={'birthdate' : 'birth_date'}, inplace=True)
lotus_evi['birth_date'] = lotus_evi['birth_date'].str.replace('/', '.')

df = player_evi.merge(lotus_evi, left_on='Player Name', right_on='Player Name', how='outer')

df["match"] = df.apply(
    lambda row: (row["birth_date_y"] in row["birth_date_x"]) or 
                (row["birth_date_x"] in row["birth_date_y"]),
    axis=1
)

print(f"Accuracy: {df['match'].mean():.2%}")