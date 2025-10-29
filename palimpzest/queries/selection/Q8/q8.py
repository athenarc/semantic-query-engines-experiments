import palimpzest as pz
from palimpzest.constants import Model
import pandas as pd
from dotenv import load_dotenv
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

model = getattr(Model, f"{args.provider.upper()}_{args.model.replace(':', '_').replace('/', '_').replace('.', '_').replace('-', '_').upper()}")

load_dotenv()

if args.wandb:
    run_name=f"palimpzest_Q8_filter_{args.model.replace(':', '_')}_{args.provider}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic selection",
    )

reports = pz.TextFileDataset(id="team_names", path="datasets/rotowire/team_names/")

reports = reports.sem_filter("The team was founded before 1970.")

config = pz.QueryProcessorConfig(
    available_models=[model],
)

output = reports.run(config=config)
output_df = output.to_df()

if args.wandb:
    if args.provider == 'ollama':
        output_file = f"evaluation/selection/Q8/results/palimpzest_Q8_filter_{args.model.replace(':', '_')}_{args.provider}.csv"
    elif args.provider == 'vllm':
        output_file = f"evaluation/selection/Q8/results/palimpzest_Q8_filter_{args.model.replace('/', '_')}_{args.provider}.csv"
    
    output_df.to_csv(output_file)
    
    wandb.log({
        "result_table": wandb.Table(dataframe=output_df),
        "execution_time": output.execution_stats.total_execution_time,
        "total_tokens": output.execution_stats.total_tokens
    })

    wandb.finish()
else:
    print("Result:\n\n", output_df)
    print("Execution time: ", output.executions_stats.total_execution_time)