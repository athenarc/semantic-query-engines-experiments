import pandas as pd
import time
import wandb
import argparse

from blendsql import BlendSQL
from blendsql.models import TransformersLLM, LiteLLM
from blendsql.ingredients import LLMQA

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=1000, const=1000, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"blendsql_Q14_aggregation_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic aggregation",
    )

df_emails = pd.read_csv(f"datasets/enron_emails/enron_emails_shuffled_{args.size}.csv")[['Subject', 'Message']]

if args.provider == 'ollama':
    model=LiteLLM(args.provider + '/' + args.model, config={"timeout": 50000}, caching=False)
elif args.provider == 'vllm':
     model = LiteLLM("hosted_vllm/" + args.model, 
                    config={"api_base": "http://localhost:5001/v1", "timeout": 50000, "cache": False}, 
                    caching=False)
db = {
    "Emails": df_emails
}

bsql = BlendSQL(
    db=db,
    model=model,
    ingredients={LLMQA}
)

start = time.time()
smoothie = bsql.execute(
    """
        SELECT {{
            LLMQA(
                'Do spam or non-spam emails prevail? from all emails? Return 1 for spam or 0 for non-spam **and only that**.',
                context=Emails.Message
            )
        }} AS Answer
    """,
    infer_gen_constraints=True,
)

exec_time = time.time()-start

if args.wandb:
    wandb.log({
        "result": wandb.Table(dataframe=smoothie.df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", smoothie.df)
    print("Execution time: ", exec_time)