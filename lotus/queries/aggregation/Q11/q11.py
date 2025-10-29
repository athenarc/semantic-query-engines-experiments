import pandas as pd
import lotus
from lotus.models import LM
import time
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=1000, const=1000, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q11_aggregation_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic aggregation",
    )

if args.provider == 'ollama':
    model = LM(args.provider + '/' + args.model)
elif args.provider == 'vllm':
    model = LM("hosted_vllm/" + args.model, api_base="http://localhost:5001/v1", api_key="dummy", timeout=50000)

lotus.settings.configure(lm=model)

df_reviews = pd.read_csv("datasets/imdb_reviews/imdb_reviews.csv").head(args.size)[['review']]

start = time.time()
df = df_reviews.sem_agg("Do positive or negative reviews prevail? from all {review}. Return 1 for positive or 0 for negative **and only that**.")
exec_time = time.time() - start

if args.wandb:
    wandb.log({
        "result": wandb.Table(dataframe=df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", df)
    print("Execution time: ", exec_time)