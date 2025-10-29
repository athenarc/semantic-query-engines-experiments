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
if args.provider == 'ollama' or args.provider == 'transformers':
    results_file = f"evaluation/projection/Q4/results/blendsql_Q4_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
elif args.provider == 'vllm':
    results_file = f"evaluation/projection/Q4/results/blendsql_Q4_map_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"

blendsql_evi = pd.read_csv(results_file)

blendsql_evi.rename(columns={'player_name': 'Player Name', '_col_1' : 'birth_date'}, inplace=True)

blendsql_evi['birth_date'] = blendsql_evi['birth_date'].str.replace('/', '.')

df = player_evi.merge(blendsql_evi, left_on='Player Name', right_on='Player Name', how='outer')

df["match"] = df.apply(
    lambda row: (
        str(row["birth_date_y"]) in str(row["birth_date_x"])
        or str(row["birth_date_x"]) in str(row["birth_date_y"])
    )
    if not (pd.isna(row["birth_date_x"]) or pd.isna(row["birth_date_y"]))
    else False,
    axis=1
)

print(f"Accuracy: {df['match'].mean():.2%}")