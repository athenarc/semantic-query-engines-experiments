import pandas as pd
import time
import wandb
import argparse

from blendsql import BlendSQL
from blendsql.models import TransformersLLM, LiteLLM
from blendsql.ingredients import LLMMap, LLMQA, LLMJoin

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=50, const=50, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"blendsql_Q1_map_{args.model.replace(':', '_').replace('/', '_')}_{args.provider}_{args.size}"

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
            'Return a list of strings with player names that did play in the game described by the given report. Please ignore the players who are mentioned and did not play but returned them all.',
            return_type='List[str]',
            Reports.Report
        )
    }}
    FROM Reports
    """,
    infer_gen_constraints=True,
)

exec_times.append(time.time()-start)


df = smoothie.df
df['player_name'] = df['_col_2'].str.replace('[', '').replace(']', '').str.split(",")
df_exploded = df.explode('player_name', ignore_index=True)
df = df_exploded.copy().drop(columns=['_col_2'])

# Points
reports = {
    "Reports" : df.copy()
}
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
        'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Game_ID, Report, player_name, {{LLMMap('How many points did the player have in the game? Return -1 if there are no mentions for points.', context, return_type='int')}} AS points
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
        'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Game_ID, Report, player_name, points, {{LLMMap('How many assists did the player have in the game? Return -1 if there are no mentions for assists.', context, return_type='int')}} AS assists
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
    'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
    FROM Reports
    ) SELECT Game_ID, Report, player_name, points, assists, {{LLMMap('How many total rebounds did the player have in the game? Return -1 if there are no mentions for total rebounds.', context, return_type='int')}} AS total_rebounds
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)

# Steals
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
        'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Game_ID, Report, player_name, points, assists, total_rebounds, {{LLMMap('How many steals did the player have in the game? Return -1 if there are no mentions for steals.', context, return_type='int')}} AS steals
    FROM joined_context
    """,
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)  

# Blocks
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
        'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
        FROM Reports
    ) SELECT Game_ID, Report, player_name, points, assists, total_rebounds, steals, {{LLMMap('How many blocks did the player have in the game? Return -1 if there are no mentions for blocks.', context, return_type='int')}} AS blocks
    FROM joined_context
    """,
    
    infer_gen_constraints=True,
)
exec_times.append(time.time() - start)

print(smoothie.df)

# # Defensive Rebounds
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, {{LLMMap('How many defensive rebounds did the player have in the game?', context)}} AS defensive_rebounds
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Offensive Rebounds
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, {{LLMMap('How many offensive rebounds did the player have in the game?', context)}} AS offensive_rebounds
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Personal Fouls
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, {{LLMMap('How many personal fouls did the player have in the game?', context)}} AS personal_fouls
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Turnovers
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, {{LLMMap('How many turnovers did the player have in the game?', context)}} AS turnovers
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Field Goals Made
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, {{LLMMap('How many field goals did the player make in the game?', context)}} AS field_goals_made
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Field Goals Attempted
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, {{LLMMap('How many field goals did the player attempt in the game?', context)}} AS field_goals_attempted
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Field Goals Percentage
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, {{LLMMap('What was the field goal percentage of player in the game?', context)}} AS field_goals_percentage
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Free Throws Made
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, field_goals_percentage, {{LLMMap('How many free throws did the player make in the game?', context)}} AS free_throws_made
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Free Throws Attempted
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, field_goals_percentage, free_throws_made, {{LLMMap('How many free throws did the player attempt in the game?', context)}} AS free_throws_attempted
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Free Throws Percentage
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, field_goals_percentage, free_throws_made, free_throws_attempted, {{LLMMap('What was the free throw percentage of player in the game?', context)}} AS free_throw_percentage
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)



# # 3-pointers made
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, field_goals_percentage, free_throws_made, free_throws_attempted, free_throw_percentage, {{LLMMap('How many 3-pointers did the player make in the game?', context)}} AS three_pointers_made
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)


# # 3-pointers attempted
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, field_goals_percentage, free_throws_made, free_throws_attempted, free_throw_percentage, three_pointers_made, {{LLMMap('How many 3-pointers did the player attempt in the game?', context)}} AS three_pointers_attempted
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # 3-pointers Percentage
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, field_goals_percentage, free_throws_made, free_throws_attempted, free_throw_percentage, three_pointers_made, three_pointers_attempted, {{LLMMap('What was the 3-pointers percentage for player in the game?', context)}} AS three_pointers_percentage
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

# # Minutes Played
# reports = {'Reports': smoothie.df }
# bsql = BlendSQL(
#     db=reports,
#     model=TransformersLLM(
#         "/data/hdd1/users/jzerv/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
#         config={"device_map": "auto"},
#         caching=False,
#     ),
#     verbose=True,
#     ingredients={LLMMap},
# )

# start = time.time()
# smoothie = bsql.execute(
#    """
#     WITH joined_context AS (
#         SELECT *,
#         'Player: ' || CAST(player_name AS VARCHAR) || '\nReport: ' || Report AS context
#         FROM Reports
#     ) SELECT Report, player_name, points, assists, total_rebounds, blocks, defensive_rebounds, offensive_rebounds, personal_fouls, turnovers, field_goals_made, field_goals_attempted, field_goals_percentage, free_throws_made, free_throws_attempted, free_throw_percentage, three_pointers_made, three_pointers_attempted, three_pointers_percentage, {{LLMMap('How many minutes did the player play in the game?', context)}} AS minutes_played
#     FROM joined_context
#     """,
#     infer_gen_constraints=True,
# )
# exec_times.append(time.time() - start)

if args.wandb:
    smoothie.df.to_csv(f"evaluation/projection/Q1/results/blendsql_Q1_map_{args.model.replace('/', '_').replace(':', '_')}_{args.provider}_{args.size}.csv")
    print("Execution time: ", sum(exec_times))
    
    wandb.log({
        "result_table": wandb.Table(dataframe=smoothie.df.fillna(-1)),
        "execution_time": sum(exec_times)
    })

    wandb.finish()
else:
    print("Result:\n\n", smoothie.df.fillna(-1))
    print("Execution time: ", sum(exec_times))

