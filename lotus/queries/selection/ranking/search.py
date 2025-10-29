import pandas as pd

import lotus
from lotus.models import LM, CrossEncoderReranker, SentenceTransformersRM

lm = LM(model="ollama/gemma3:12b")
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
reranker = CrossEncoderReranker(model="mixedbread-ai/mxbai-rerank-large-v1")

lotus.settings.configure(lm=lm, rm=rm, reranker=reranker)
df_players = pd.read_csv("../../datasets/reports_with_players_100.csv")
df_players = df_players[df_players["Game ID"] < 14] # Keep total of ~100 entries
df_players = df_players[['Game ID', 'Report']]

df = df_players.sem_index("Report", "sem_index").sem_searcH("Report", "Which report is referring to Bucks?", K=4, n_rerank=2)

print(df)