import pandas as pd
import lotus
from lotus.models import LM
import os
import wandb
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=50, const=50, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q2_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

    wandb.init(
        project="semantic_operations",
        name=run_name,
        group="semantic derivation",
    )

if args.provider == 'ollama':
    lm = LM(args.provider + '/' + args.model)
elif args.provider == 'vllm':
    lm = LM("hosted_vllm/" + args.model, api_base="http://localhost:5001/v1", api_key="dummy", timeout=50000)

lotus.settings.configure(lm=lm)
df_reports = pd.read_csv("datasets/rotowire/reports_table.csv").head(args.size).rename(columns={'Game_ID' : 'Game ID'})
elapsed_times = []

# Retrieve team names from the reports
examples = {
    "Report": ["The Milwaukee Bucks (18 - 17) defeated the New York Knicks (5 - 31) 95 - 82 on Sunday at Madison Square Garden in New York. The Bucks were able to have a great night defensively, giving themselves the scoring advantage in all four quarters. The Bucks showed superior shooting, going 46 percent from the field, while the Knicks went only 41 percent from the floor. The Bucks also out - rebounded the Knicks 48 - 36, giving them in an even further advantage which helped them secure the 13 - point victory on the road. Brandon Knight led the Bucks again in this one. He went 6 - for - 14 from the field and 1 - for - 3 from beyond the arc to score 17 points, while also handing out five assists. He's now averaging 21 points per game over his last three games, as he's consistently been the offensive leader for this team. Zaza Pachulia also had a strong showing, finishing with 16 points (6 - 12 FG, 4 - 4 FT) and a team - high of 14 rebounds. It marked his second double - double in a row and fourth on the season, as the inexperienced centers on the Knicks' roster were n't able to limit him. Notching a double - double of his own, Giannis Antetokounmpo recorded 16 points (6 - 9 FG, 1 - 1 3Pt, 3 - 6 FT) and 12 rebounds. The 12 rebounds matched a season - high, while it was his second double - double of the season. Coming off the bench for a big night was Kendall Marshall. He went 6 - for - 8 from the field and 3 - for - 3 from the free throw line to score 15 points in 20 minutes. The Knicks really struggled to score without Carmelo Anthony and Amare Stoudemire. Tim Hardaway Jr led the team as the starting shooting guard, going 6 - for - 13 from the field and 3 - for - 5 from the three - point line to score 17 points, while also adding four assists. He's now scored 17 or more points in three out of his last four games, as he has put it on himself to pick up the slack with other key players sitting out. J.R. Smith also put together a solid outing as a starter. He finished with 15 points and seven rebounds in 37 minutes. Like Haradaway Jr, he's also benefitted from other guys sitting out, and has now combined for 37 points over his last two games. While he did n't have his best night defensively, Cole Aldrich scored 12 points (6 - 10 FG) and grabbed seven rebounds in 19 minutes. The only other Knick to reach double figures in points was Jason Smith, who came off the bench for 10 points (3 - 11 FG, 4 - 4 FT). The Bucks' next game will be at home against the Phoenix Suns on Tuesday, while the Knicks will travel to Memphis to play the Grizzlies on Monday."],
    "Answer": ["Milwaukee Bucks, New York Knicks"]
}
examples_df = pd.DataFrame(examples)

user_instruction = "What are the teams that played in the game {Report}? Please list them in a comma-separated format."
start = time.time()
df = df_reports.sem_map(user_instruction, examples=examples_df)
end = time.time()
elapsed_times.append(end-start)

df_players = df[['Game ID', '_map']].copy()

df_players['team_name'] = df_players['_map'].str.split(", ")

df_exploded = df_players.explode('team_name', ignore_index=True)

df_players = df_exploded[['Game ID', 'team_name']].copy()

df_merged = pd.merge(df_players, df[['Game ID', 'Report']], on='Game ID', how='left')

# Extract Wins
print("Extracting Wins")
examples = {
    "team_name": ["Milwaukee Bucks"],
    "Report": ["The Milwaukee Bucks (18 - 17) defeated the New York Knicks (5 - 31) 95 - 82 on Sunday at Madison Square Garden in New York. The Bucks were able to have a great night defensively, giving themselves the scoring advantage in all four quarters. The Bucks showed superior shooting, going 46 percent from the field, while the Knicks went only 41 percent from the floor. The Bucks also out - rebounded the Knicks 48 - 36, giving them in an even further advantage which helped them secure the 13 - point victory on the road. Brandon Knight led the Bucks again in this one. He went 6 - for - 14 from the field and 1 - for - 3 from beyond the arc to score 17 points, while also handing out five assists. He's now averaging 21 points per game over his last three games, as he's consistently been the offensive leader for this team. Zaza Pachulia also had a strong showing, finishing with 16 points (6 - 12 FG, 4 - 4 FT) and a team - high of 14 rebounds. It marked his second double - double in a row and fourth on the season, as the inexperienced centers on the Knicks' roster were n't able to limit him. Notching a double - double of his own, Giannis Antetokounmpo recorded 16 points (6 - 9 FG, 1 - 1 3Pt, 3 - 6 FT) and 12 rebounds. The 12 rebounds matched a season - high, while it was his second double - double of the season. Coming off the bench for a big night was Kendall Marshall. He went 6 - for - 8 from the field and 3 - for - 3 from the free throw line to score 15 points in 20 minutes. The Knicks really struggled to score without Carmelo Anthony and Amare Stoudemire. Tim Hardaway Jr led the team as the starting shooting guard, going 6 - for - 13 from the field and 3 - for - 5 from the three - point line to score 17 points, while also adding four assists. He's now scored 17 or more points in three out of his last four games, as he has put it on himself to pick up the slack with other key players sitting out. J.R. Smith also put together a solid outing as a starter. He finished with 15 points and seven rebounds in 37 minutes. Like Haradaway Jr, he's also benefitted from other guys sitting out, and has now combined for 37 points over his last two games. While he did n't have his best night defensively, Cole Aldrich scored 12 points (6 - 10 FG) and grabbed seven rebounds in 19 minutes. The only other Knick to reach double figures in points was Jason Smith, who came off the bench for 10 points (3 - 11 FG, 4 - 4 FT). The Bucks' next game will be at home against the Phoenix Suns on Tuesday, while the Knicks will travel to Memphis to play the Grizzlies on Monday."],
    "Answer": [18]
}
user_instruction = "What is the number of wins for {team_name} in the game {Report} if they are mentioned. Return only the number (only an integer) of Wins or -1 if there are no mentions for Wins for {team_name} (without explanation)."
start = time.time()
df_wins = df_merged.sem_map(user_instruction)
end = time.time()
elapsed_times.append(end-start)
print(" Extraction, Elapsed Time: ", end-start, '\n')
df = df_wins.rename(columns={"_map": "wins"})

# Extract Losses
print("Extracting Losses")
examples = {
    "team_name": ["Milwaukee Bucks"],
    "Report": ["The Milwaukee Bucks (18 - 17) defeated the New York Knicks (5 - 31) 95 - 82 on Sunday at Madison Square Garden in New York. The Bucks were able to have a great night defensively, giving themselves the scoring advantage in all four quarters. The Bucks showed superior shooting, going 46 percent from the field, while the Knicks went only 41 percent from the floor. The Bucks also out - rebounded the Knicks 48 - 36, giving them in an even further advantage which helped them secure the 13 - point victory on the road. Brandon Knight led the Bucks again in this one. He went 6 - for - 14 from the field and 1 - for - 3 from beyond the arc to score 17 points, while also handing out five assists. He's now averaging 21 points per game over his last three games, as he's consistently been the offensive leader for this team. Zaza Pachulia also had a strong showing, finishing with 16 points (6 - 12 FG, 4 - 4 FT) and a team - high of 14 rebounds. It marked his second double - double in a row and fourth on the season, as the inexperienced centers on the Knicks' roster were n't able to limit him. Notching a double - double of his own, Giannis Antetokounmpo recorded 16 points (6 - 9 FG, 1 - 1 3Pt, 3 - 6 FT) and 12 rebounds. The 12 rebounds matched a season - high, while it was his second double - double of the season. Coming off the bench for a big night was Kendall Marshall. He went 6 - for - 8 from the field and 3 - for - 3 from the free throw line to score 15 points in 20 minutes. The Knicks really struggled to score without Carmelo Anthony and Amare Stoudemire. Tim Hardaway Jr led the team as the starting shooting guard, going 6 - for - 13 from the field and 3 - for - 5 from the three - point line to score 17 points, while also adding four assists. He's now scored 17 or more points in three out of his last four games, as he has put it on himself to pick up the slack with other key players sitting out. J.R. Smith also put together a solid outing as a starter. He finished with 15 points and seven rebounds in 37 minutes. Like Haradaway Jr, he's also benefitted from other guys sitting out, and has now combined for 37 points over his last two games. While he did n't have his best night defensively, Cole Aldrich scored 12 points (6 - 10 FG) and grabbed seven rebounds in 19 minutes. The only other Knick to reach double figures in points was Jason Smith, who came off the bench for 10 points (3 - 11 FG, 4 - 4 FT). The Bucks' next game will be at home against the Phoenix Suns on Tuesday, while the Knicks will travel to Memphis to play the Grizzlies on Monday."],
    "Answer": [17]
}
user_instruction = "What is the number of Losses for {team_name} in the game {Report} if they are mentioned. Return only the number (only an integer) of Losses or -1 if there are no mentions for Losses for {team_name} (without explanation)."
start = time.time()
df_losses = df.sem_map(user_instruction)
end = time.time()
elapsed_times.append(end-start)
print(" Extraction, Elapsed Time: ", end-start, '\n')
df = df_losses.rename(columns={"_map": "losses"})

# Extract Total Points
print("Extracting Total Points")
examples = {
    "team_name": ["Milwaukee Bucks"],
    "Report": ["The Milwaukee Bucks (18 - 17) defeated the New York Knicks (5 - 31) 95 - 82 on Sunday at Madison Square Garden in New York. The Bucks were able to have a great night defensively, giving themselves the scoring advantage in all four quarters. The Bucks showed superior shooting, going 46 percent from the field, while the Knicks went only 41 percent from the floor. The Bucks also out - rebounded the Knicks 48 - 36, giving them in an even further advantage which helped them secure the 13 - point victory on the road. Brandon Knight led the Bucks again in this one. He went 6 - for - 14 from the field and 1 - for - 3 from beyond the arc to score 17 points, while also handing out five assists. He's now averaging 21 points per game over his last three games, as he's consistently been the offensive leader for this team. Zaza Pachulia also had a strong showing, finishing with 16 points (6 - 12 FG, 4 - 4 FT) and a team - high of 14 rebounds. It marked his second double - double in a row and fourth on the season, as the inexperienced centers on the Knicks' roster were n't able to limit him. Notching a double - double of his own, Giannis Antetokounmpo recorded 16 points (6 - 9 FG, 1 - 1 3Pt, 3 - 6 FT) and 12 rebounds. The 12 rebounds matched a season - high, while it was his second double - double of the season. Coming off the bench for a big night was Kendall Marshall. He went 6 - for - 8 from the field and 3 - for - 3 from the free throw line to score 15 points in 20 minutes. The Knicks really struggled to score without Carmelo Anthony and Amare Stoudemire. Tim Hardaway Jr led the team as the starting shooting guard, going 6 - for - 13 from the field and 3 - for - 5 from the three - point line to score 17 points, while also adding four assists. He's now scored 17 or more points in three out of his last four games, as he has put it on himself to pick up the slack with other key players sitting out. J.R. Smith also put together a solid outing as a starter. He finished with 15 points and seven rebounds in 37 minutes. Like Haradaway Jr, he's also benefitted from other guys sitting out, and has now combined for 37 points over his last two games. While he did n't have his best night defensively, Cole Aldrich scored 12 points (6 - 10 FG) and grabbed seven rebounds in 19 minutes. The only other Knick to reach double figures in points was Jason Smith, who came off the bench for 10 points (3 - 11 FG, 4 - 4 FT). The Bucks' next game will be at home against the Phoenix Suns on Tuesday, while the Knicks will travel to Memphis to play the Grizzlies on Monday."],
    "Answer": [95]
}
user_instruction = "What is the number of total points for {team_name} in the game {Report} if they are mentioned. Return only the number (only an integer) of total points or -1 if there are no mentions for total points for {team_name} (without explanation)."
start = time.time()
df_total_points = df.sem_map(user_instruction)
end = time.time()
elapsed_times.append(end-start)
print(" Extraction, Elapsed Time: ", end-start, '\n')
df = df_total_points.rename(columns={"_map": "total_points"})

exec_time = sum(elapsed_times)

if args.wandb:
    if args.provider == 'ollama':
        output_file = f"evaluation/derivation/Q2/results/lotus_Q2_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
    elif args.provider =='vllm':
        output_file = f"evaluation/derivation/Q2/results/lotus_Q2_map_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"
        
    df.to_csv(output_file)

    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", df)
    print("Execution time: ", exec_time)