import palimpzest as pz
from palimpzest.constants import Model
import pandas as pd
from dotenv import load_dotenv
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

model = getattr(Model, f"{args.provider.upper()}_{args.model.replace(':', '_').replace('/', '_').replace('.', '_').replace('-', '_').upper()}")

load_dotenv()

# UDF to create one record for each player in the list of player names
def explode_player_list(record: dict):
    team_name_list = record.get("team_name_list") or []
    team_name_list = filter(None, team_name_list)
    records = []

    for team_name in team_name_list:
        out_record = {k: v for k, v in record.items()}
        out_record['team_name'] = team_name
        records.append(out_record)
    return records

if args.wandb:
    run_name=f"palimpzest_Q2_project_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic derivation",
    )

# updated PZ program
reports = pz.TextFileDataset(id="rotowire_reports", path=f"datasets/rotowire/reports/{args.size}/")
reports = reports.sem_add_columns([
    {"name": "team_name_list", "type": list[str], "desc": "Names of teams who played the game, excluding those who are mentioned but did not play."},
])


reports = reports.add_columns(
    udf=explode_player_list,
    cols=[{"name": "team_name", "type": str, "desc": "The name of an NBA team who played in a given game."}],
    cardinality=pz.Cardinality.ONE_TO_MANY,
)
reports = reports.sem_add_columns(
    cols=[
        {"name": "Wins", "type": int, "desc": "The number of Wins that `team_name` has. If the team's Wins are not mentioned in the report, fill the value with -1."},
        {"name": "Losses", "type": int, "desc": "The number of Losses that `team_name` has. If the team's Losses are not mentioned in the report, fill the value with -1."},
        {"name": "Total Points", "type": int, "desc": "The number of Total Points that `team_name` scored. If the team's Total Points are not mentioned in the report, fill the value with -1."},
    ],
    depends_on=["contents", "team_name"],
)
config = pz.QueryProcessorConfig(
    available_models=[model],
    timeout=50000
)

output = reports.run(config=config)
output_df = output.to_df()
output_df.drop(columns=["team_name_list"], inplace=True)

if args.wandb:
    if args.provider == 'ollama':
        output_file = f"evaluation/derivation/Q2/results/palimpzest_Q2_project_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
    elif args.provider == 'vllm':
        output_file = f"evaluation/derivation/Q2/results/palimpzest_Q2_project_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"
    
    output_df.to_csv(output_file)
    wandb.log({
        "result_table": wandb.Table(dataframe=output_df),
        "execution_time": output.execution_stats.total_execution_time,
        "total_tokens": output.execution_stats.total_tokens
    })

    wandb.finish()
else:
    print("Result:\n\n", output_df)
    print("Execution time: ", output.execution_stats.total_execution_time)