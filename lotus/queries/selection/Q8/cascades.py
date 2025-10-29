import pandas as pd
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs
import time
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q8_filter_cascades_{args.model.replace(':', '_')}_{args.provider}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic selection",
    )

if (args.provider == 'ollama'):
    lm = LM(args.provider + '/' + args.model)
elif args.provider == 'vllm':
    lm = LM("hosted_vllm/" + args.model, api_base="http://localhost:5001/v1", api_key="dummy", timeout=50000)

helper_lm = LM(model="hosted_vllm/meta-llama/Llama-3.1-8B-Instruct", api_base="http://localhost:5001/v1", api_key="dummy")
lotus.settings.configure(lm=lm, helper_lm=helper_lm)

df_teams = pd.read_csv("datasets/rotowire/team_evidence.csv")
df_teams = pd.DataFrame(df_teams['Team Name']).rename(columns={'Team Name' : 'team_name'})

user_instruction = "{team_name} founded before 1970."

cascade_args = CascadeArgs(recall_target=0.9, precision_target=0.9, sampling_percentage=0.2, failure_probability=0.1)

start = time.time()
df = df_teams.sem_filter(user_instruction, cascade_args=cascade_args)
exec_time = time.time() - start

if args.wandb:
    if args.provider == 'ollama':
        output_file = f"evaluation/selection/Q8/results/lotus_Q8_filter_cascades_{args.model.replace(':', '_')}_{args.provider}.csv"
    elif args.provider =='vllm':
        output_file = f"evaluation/selection/Q8/results/lotus_Q8_filter_cascades_{args.model.replace('/', '_')}_{args.provider}.csv"
        
    df.to_csv(output_file)

    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", df)
    print("Execution time: ", exec_time)