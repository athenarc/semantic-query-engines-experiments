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
    player_name_list = record.get("player_name_list") or []
    player_name_list = filter(None, player_name_list)
    records = []

    for player_name in player_name_list:
        out_record = {k: v for k, v in record.items()}
        out_record['player_name'] = player_name
        records.append(out_record)
    return records

if args.wandb:
    run_name=f"palimpzest_Q1_project_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic derivation",
)

reports = pz.TextFileDataset(id="rotowire_reports", path=f"datasets/rotowire/reports/{args.size}/")
reports = reports.sem_add_columns([
    {"name": "player_name_list", "type": list[str], "desc": "Names of players who played the game, excluding those who are mentioned but did not play."},
])


reports = reports.add_columns(
    udf=explode_player_list,
    cols=[{"name": "player_name", "type": str, "desc": "The name of an NBA player who played in a given game."}],
    cardinality=pz.Cardinality.ONE_TO_MANY,
)
reports = reports.sem_add_columns(
    cols=[
        {"name": "Points", "type": float, "desc": "The points per game for the player specified by the `player_name` field. If the player's points are not mentioned in the report, fill the value with -1."},
        {"name": "Assists", "type": float, "desc": "The assists per game for the player specified by the `player_name` field. If the player's assists are not mentioned in the report, fill the value with -1."},
        {"name": "Total Rebounds", "type": float, "desc": "The total number of rebounds per game for the player specified by the `player_name` field. If the player's total number of rebounds are not mentioned in the report, fill the value with -1."},
        {"name": "Steals", "type": float, "desc": "The steals per game for the player specified by the `player_name` field. If the player's steals are not mentioned in the report, fill the value with -1."},
        {"name": "Blocks", "type": float, "desc": "The blocks per game for the player specified by the `player_name` field. If the player's blocks are not mentioned in the report, fill the value with -1."},
        # {"name": "Defensive Rebounds", "type": float, "desc": "The defensive rebounds per game for the player specified by the `player_name` field. If the player's defensive rebounds are not mentioned in the report, fill the value with -1."},
        # {"name": "Offensive Rebounds", "type": float, "desc": "The offensive rebounds per game for the player specified by the `player_name` field. If the player's offensive rebounds are not mentioned in the report, fill the value with -1."},
        # {"name": "Personal Fouls", "type": float, "desc": "The personal fouls per game for the player specified by the `player_name` field. If the player's personal fouls are not mentioned in the report, fill the value with -1."},
        # {"name": "Turnovers", "type": float, "desc": "The turnovers per game for the player specified by the `player_name` field. If the player's turnovers are not mentioned in the report, fill the value with -1."},
        # {"name": "Field Goals Made", "type": float, "desc": "The field goals made per game for the player specified by the `player_name` field. If the player's field goals made are not mentioned in the report, fill the value with -1."},
        # {"name": "Field Goals Attempted", "type": float, "desc": "The field goals attempted per game for the player specified by the `player_name` field. If the player's field goals attempted are not mentioned in the report, fill the value with -1."},
        # {"name": "Field Goal Percentage", "type": float, "desc": "The field goal percentage per game for the player specified by the `player_name` field. If the player's field goal percentage is not mentioned in the report, fill the value with -1."},
        # {"name": "Free Throws Made", "type": float, "desc": "The free throws made per game for the player specified by the `player_name` field. If the player's free throws made are not mentioned in the report, fill the value with -1."},
        # {"name": "Free Throws Attempted", "type": float, "desc": "The free throws attempted per game for the player specified by the `player_name` field. If the player's free throws attempted are not mentioned in the report, fill the value with -1."},
        # {"name": "Free Throw Percentage", "type": float, "desc": "The free throw percentage per game for the player specified by the `player_name` field. If the player's free throw percentage is not mentioned in the report, fill the value with -1."},
        # {"name": "Three Pointers Made", "type": float, "desc": "The three pointers made per game for the player specified by the `player_name` field. If the player's three pointers made are not mentioned in the report, fill the value with -1."},
        # {"name": "Three Pointers Attempted", "type": float, "desc": "The three pointers attempted per game for the player specified by the `player_name` field. If the player's three pointers attempted are not mentioned in the report, fill the value with -1."},
        # {"name": "Three Point Percentage", "type": float, "desc": "The three point field goal percentage per game for the player specified by the `player_name` field. If the player's three point field goal percentage is not mentioned in the report, fill the value with -1."},
        # {"name": "Minutes Played", "type": str, "desc": "The minutes played per game for the player specified by the `player_name` field. If the player's minutes played are not mentioned in the report, fill the value with -1."},
    ],
    depends_on=["contents", "player_name"],
)
config = pz.QueryProcessorConfig(
    available_models=[model],
    timeout=50000,
)

output = reports.run(config=config)
output_df = output.to_df()
output_df.drop(columns=["player_name_list"], inplace=True)

if args.wandb:
    if args.provider == 'ollama':
        output_file = f"evaluation/derivation/Q1/results/palimpzest_Q1_project_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
    elif args.provider == 'vllm':
        output_file = f"evaluation/derivation/Q1/results/palimpzest_Q1_project_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"
    
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