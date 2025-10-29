import pandas as pd
import lotus
from lotus.models import LM
import time
import wandb

lm = LM(model="ollama/gemma3:12b")

lotus.settings.configure(lm=lm)
df_players = pd.read_csv("../../datasets/reports_with_players_100.csv")
df_players = df_players[df_players["Game ID"] < 14] # Keep total of ~100 entries
df_players = df_players.rename(columns={'Player Name' : 'player_name'})

sorted_df, stats = df_players.sem_topk("Which {player_name} scored the most points in the games described by {Report}?", K=3, return_stats=True)

print(sorted_df)