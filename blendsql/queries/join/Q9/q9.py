import pandas as pd
import time
import wandb
import argparse

from blendsql import BlendSQL
from blendsql.models import TransformersLLM, LiteLLM
from blendsql.ingredients import LLMJoin

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=50, const=50, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"blendsql_Q9_join_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic join",
    )

players_df = pd.read_csv('datasets/rotowire/player_evidence_mine.csv').head(args.size)[['Player Name']].rename(columns={'Player Name' : 'player_name'})

teams_df = pd.read_csv('datasets/rotowire/team_evidence.csv')[['Team Name']].rename(columns={'Team Name' : 'team_name'})

db = {
    "Players": pd.DataFrame(players_df),
    "Teams": pd.DataFrame(teams_df)
}

if args.provider == 'ollama':
    model = LiteLLM(args.provider + '/' + args.model, 
                    config={"timeout" : 50000, "cache": False},
                    caching=False)
elif args.provider == 'vllm':
    model = LiteLLM("hosted_vllm/" + args.model, 
                    config={"api_base": "http://localhost:5001/v1", "timeout": 50000, "cache": False}, 
                    caching=False)

bsql = BlendSQL(
    db=db,
    model=model,
    ingredients={LLMJoin}
)

start = time.time()
smoothie = bsql.execute(
    """
        SELECT *
        FROM Players p
        JOIN Teams t ON {{
            LLMJoin(
                p.player_name,
                t.team_name,
                join_criteria='The player was playing for the team in 2015.',
            )
        }} 
    """,
    infer_gen_constraints=True,
)

exec_time = time.time()-start

if args.wandb:
    smoothie.df.to_csv(f"evaluation/join/Q9/results/blendsql_Q9_join_{args.model.replace('/', ':')}_{args.provider}_{args.size}.csv")

    wandb.log({
        "result_table": wandb.Table(dataframe=smoothie.df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", smoothie.df)
    print("Execution time: ", exec_time)

