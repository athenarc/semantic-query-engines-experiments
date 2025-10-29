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
    run_name = f"blendsql_Q2_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic projection",

)

# Load reports dataset
reports_table = pd.read_csv('datasets/rotowire/reports_table.csv').head(args.size)
reports = {
    "Reports" : pd.DataFrame(reports_table)
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
    db=reports,
    model=model,
    verbose=True,
    ingredients={LLMMap},
)

exec_times = []
start = time.time()

smoothie = bsql.execute(
   """
    SELECT Game_ID, Reports.Report, {{
        LLMMAP(
            'Return a list of strings with team names that did play in the game. Please ignore the teams who are mentioned and did not play.',
            Reports.Report,
            return_type='List[str]'
        )
    }}
    FROM Reports
    """,
    infer_gen_constraints=True,
)

exec_times.append(time.time()-start)

df = smoothie.df
df['team_name'] = df['_col_2'].str.split(",")
df_exploded = df.explode('team_name', ignore_index=True)
df = df_exploded.copy().drop(columns=['_col_2'])

# Points
reports = {'Reports': df.copy() }
bsql = BlendSQL(
    db=reports,
    model=model,
    verbose=True,
    ingredients={LLMMap},
)

start = time.time()
smoothie = bsql.execute(
   """
    WITH joined_context AS (
        SELECT *,
        'Team: ' || CAST(team_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Game_ID, Report, team_name, {{LLMMap('How many Wins has the team?', context, return_type='int')}} AS wins
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)

# Assists
reports = { 'Reports': smoothie.df }
bsql = BlendSQL(
    db=reports,
    model=model,
    verbose=True,
    ingredients={LLMMap},
)

start = time.time()
smoothie = bsql.execute(
   """
    WITH joined_context AS (
        SELECT *,
        'Team: ' || CAST(team_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Game_ID, Report, team_name, wins, {{LLMMap('How many Losses has the team?', context, return_type='int')}} AS losses
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)

# Total Rebounds
reports = {'Reports': smoothie.df }
bsql = BlendSQL(
    db=reports,
    model=model,
    verbose=True,
    ingredients={LLMMap},
)

start = time.time()
smoothie = bsql.execute(
   """
    WITH joined_context AS (
    SELECT *,
    'Team: ' || CAST(team_name AS VARCHAR) || '\nReport: ' || Report AS context
    FROM Reports
    ) SELECT Game_ID, Report, team_name, wins, losses, {{LLMMap('What is the number of total points the team scored?', context, return_type='int')}} AS total_points
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)

if args.wandb:
    smoothie.df.to_csv(f"evaluation/projection/Q2/results/blendsql_Q2_map_{args.model.replace('/', '_').replace(':', '_')}_{args.provider}_{args.size}.csv")
    print("Execution time: ", sum(exec_times))
    
    wandb.log({
        "result_table": wandb.Table(dataframe=smoothie.df.fillna("-1")),
        "execution_time": sum(exec_times)
    })

    wandb.finish()
else:
    print("Result:\n\n", smoothie.df)
    print("Execution time: ", sum(exec_times))


