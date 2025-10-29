import pandas as pd
import lotus
from lotus.models import LM
import wandb
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=50, const=50, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q9_join_default_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic join",
    )

df_players = pd.read_csv("datasets/rotowire/player_evidence_mine.csv").head(args.size)[['Player Name']]
df_teams = pd.read_csv("datasets/rotowire/team_evidence.csv")[['Team Name']]

if args.provider == 'ollama':
    model = LM(args.provider + '/' + args.model)
elif args.provider == 'vllm':
    model = LM("hosted_vllm/" + args.model, api_base="http://localhost:5001/v1", api_key="dummy", timeout=50000)

lotus.settings.configure(lm=model)

instruction = "The player {Player Name:left} was playing for team {Team Name:right} in 2015."
start = time.time()
df = df_players.sem_join(df_teams, instruction)
exec_time = time.time() - start

if args.wandb:
    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", df)
    print("Execution time: ", exec_time)