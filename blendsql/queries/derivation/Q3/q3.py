import pandas as pd
import time
import wandb
import argparse

from blendsql import BlendSQL
from blendsql.models import TransformersLLM, LiteLLM
from blendsql.ingredients import LLMMap, LLMQA, LLMJoin

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"blendsql_Q3_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic derivation",
)

# Load reports
reports = pd.read_csv('datasets/rotowire/player_evidence_mine.csv').dropna(subset=['nationality']).head(args.size)
reports.rename(columns={"Player Name": "player_name"}, inplace=True)
players = {
    "Players" : pd.DataFrame(reports['player_name'])
}

if args.provider == 'ollama':
    model = LiteLLM(args.provider + '/' + args.model, 
                    config={"timeout" : 50000, "cache": False},
                    caching=False)
elif args.provider == 'vllm':
    model = LiteLLM("hosted_vllm/" + args.model, 
                    config={"api_base": "http://localhost:5001/v1", "timeout": 50000, "cache": False}, 
                    caching=False)
elif args.provider == 'transformers':
    model = TransformersLLM(
        "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        config={"device_map": "auto"},
        caching=False,
    )


# Prepare our BlendSQL connection
bsql = BlendSQL(
    db=players,
    model=model,
    verbose=True,
    ingredients={LLMMap},
)

start = time.time()

smoothie = bsql.execute(
   """
    SELECT Players.player_name, {{
        LLMMAP(
            'Return the nationality of the player.',
            return_type='str',
            Players.player_name,
        )
    }}
    FROM Players
    """,
    infer_gen_constraints=True,
)

exec_time = time.time() - start
print(smoothie.df)

if args.wandb:
    smoothie.df.to_csv(f"evaluation/derivation/Q3/results/blendsql_Q3_map_{args.model.replace('/', '_').replace(':', '_')}_{args.provider}_{args.size}.csv")

    wandb.log({
        "result_table": wandb.Table(dataframe=smoothie.df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", smoothie.df)
    print("Execution time: ", exec_time)