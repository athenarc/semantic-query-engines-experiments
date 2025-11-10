import pandas as pd
import lotus
from lotus.models import LM
import os
import wandb
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q2_extract_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic derivation",
    )

if args.provider == 'ollama':
    lm = LM(args.provider + '/' + args.model)
elif args.provider == 'vllm':
    lm = LM("hosted_vllm/" + args.model, api_base="http://localhost:5001/v1", api_key="dummy", timeout=50000)

lotus.settings.configure(lm=lm)
df_reports = pd.read_csv("datasets/rotowire/reports_table.csv").head(args.size).rename(columns={'Game_ID' : 'Game ID'})

input_cols = ["Report"]

start = time.time()
# A description can be specified for each output column
output_cols = {
    "masked": "A comma-separated list with team names that played in the game. Do not count teams that are mentioned but did not play.",
}

new_df = df_reports.sem_extract(input_cols, output_cols) 

df_players = new_df[['Game ID', 'masked']].copy()

df_players['team_name'] = df_players['masked'].str.split(", ")

df_exploded = df_players.explode('team_name', ignore_index=True)

df_players = df_exploded[['Game ID', 'team_name']].copy()

df_merged = pd.merge(df_players, new_df[['Game ID', 'Report']], on='Game ID', how='left')

input_cols = ["Report", "team_name"]
output_cols = {
    "masked_col2": "The number of Wins that the {team_name} has or -1 if not mentioned.",
    "masked_col3": "The number of Losses that the {team_name} has or -1 if not mentioned.",
    "masked_col4": "The number of Total Points that the {team_name} scored or -1 if not mentioned",
}
new_df = df_merged.sem_extract(input_cols, output_cols, extract_quotes=False)

new_df = new_df.rename(columns={"masked_col2": "wins", "masked_col3": "losses", "masked_col4": "total_points"})
df = new_df[['Game ID', 'team_name', 'wins', 'losses', 'total_points']]
exec_time = time.time() - start

if args.wandb:
    if args.provider == 'ollama':
        output_file = f"evaluation/derivation/Q2/results/lotus_Q2_extract_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
    elif args.provider =='vllm':
        output_file = f"evaluation/derivation/Q2/results/lotus_Q2_extract_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"
        
    df.to_csv(output_file)

    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", df)
    print("Execution time: ", exec_time)