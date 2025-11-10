import pandas as pd
import lotus
from lotus.models import LM
from time import time
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action='store_true', help="Enables wandb report")
parser.add_argument("-s", "--size", nargs='?', default=100, const=100, type=int, help="The input size")
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

if args.wandb:
    run_name = f"lotus_Q1_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}"

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

# Retrieve player names from the reports
examples = {
    "Report": ["The Milwaukee Bucks (18 - 17) defeated the New York Knicks (5 - 31) 95 - 82 on Sunday at Madison Square Garden in New York. The Bucks were able to have a great night defensively, giving themselves the scoring advantage in all four quarters. The Bucks showed superior shooting, going 46 percent from the field, while the Knicks went only 41 percent from the floor. The Bucks also out - rebounded the Knicks 48 - 36, giving them in an even further advantage which helped them secure the 13 - point victory on the road. Brandon Knight led the Bucks again in this one. He went 6 - for - 14 from the field and 1 - for - 3 from beyond the arc to score 17 points, while also handing out five assists. He's now averaging 21 points per game over his last three games, as he's consistently been the offensive leader for this team. Zaza Pachulia also had a strong showing, finishing with 16 points (6 - 12 FG, 4 - 4 FT) and a team - high of 14 rebounds. It marked his second double - double in a row and fourth on the season, as the inexperienced centers on the Knicks' roster were n't able to limit him. Notching a double - double of his own, Giannis Antetokounmpo recorded 16 points (6 - 9 FG, 1 - 1 3Pt, 3 - 6 FT) and 12 rebounds. The 12 rebounds matched a season - high, while it was his second double - double of the season. Coming off the bench for a big night was Kendall Marshall. He went 6 - for - 8 from the field and 3 - for - 3 from the free throw line to score 15 points in 20 minutes. The Knicks really struggled to score without Carmelo Anthony and Amare Stoudemire. Tim Hardaway Jr led the team as the starting shooting guard, going 6 - for - 13 from the field and 3 - for - 5 from the three - point line to score 17 points, while also adding four assists. He's now scored 17 or more points in three out of his last four games, as he has put it on himself to pick up the slack with other key players sitting out. J.R. Smith also put together a solid outing as a starter. He finished with 15 points and seven rebounds in 37 minutes. Like Haradaway Jr, he's also benefitted from other guys sitting out, and has now combined for 37 points over his last two games. While he did n't have his best night defensively, Cole Aldrich scored 12 points (6 - 10 FG) and grabbed seven rebounds in 19 minutes. The only other Knick to reach double figures in points was Jason Smith, who came off the bench for 10 points (3 - 11 FG, 4 - 4 FT). The Bucks' next game will be at home against the Phoenix Suns on Tuesday, while the Knicks will travel to Memphis to play the Grizzlies on Monday."],
    "Answer": ["Brandon Knight, Zaza Pachulia, Giannis Antetokounmpo, Kendall Marshall, Tim Hardaway Jr, J.R. Smith, Cole Aldrich, Jason Smith"]
}
examples_df = pd.DataFrame(examples)

user_instruction = "What are the players that played in the game {Report}? Please list them in a comma-separated format and do not count players that didn't play."
start = time()
df = df_reports.sem_map(user_instruction, examples=examples_df)
end = time()
elapsed_times.append(end-start)
print("Player Name Extraction, Elapsed Time: ", end-start)

# Post processing step
df_players = df[['Game ID', '_map']].copy()

df_players['player_name']=  df_players['_map'].str.split(", ")

df_exploded = df_players.explode('player_name', ignore_index=True)

df_players = df_exploded[['Game ID', 'player_name']].copy()

df_merged = pd.merge(df_players, df_reports[['Game ID', 'Report']], on='Game ID', how='left')

# Extract assists
print("Extracting Assists")
examples = {
    "player_name": ["Tim Hardaway Jr"],
    "Report": ["The Milwaukee Bucks (18 - 17) defeated the New York Knicks (5 - 31) 95 - 82 on Sunday at Madison Square Garden in New York. The Bucks were able to have a great night defensively, giving themselves the scoring advantage in all four quarters. The Bucks showed superior shooting, going 46 percent from the field, while the Knicks went only 41 percent from the floor. The Bucks also out - rebounded the Knicks 48 - 36, giving them in an even further advantage which helped them secure the 13 - point victory on the road. Brandon Knight led the Bucks again in this one. He went 6 - for - 14 from the field and 1 - for - 3 from beyond the arc to score 17 points, while also handing out five assists. He's now averaging 21 points per game over his last three games, as he's consistently been the offensive leader for this team. Zaza Pachulia also had a strong showing, finishing with 16 points (6 - 12 FG, 4 - 4 FT) and a team - high of 14 rebounds. It marked his second double - double in a row and fourth on the season, as the inexperienced centers on the Knicks' roster were n't able to limit him. Notching a double - double of his own, Giannis Antetokounmpo recorded 16 points (6 - 9 FG, 1 - 1 3Pt, 3 - 6 FT) and 12 rebounds. The 12 rebounds matched a season - high, while it was his second double - double of the season. Coming off the bench for a big night was Kendall Marshall. He went 6 - for - 8 from the field and 3 - for - 3 from the free throw line to score 15 points in 20 minutes. The Knicks really struggled to score without Carmelo Anthony and Amare Stoudemire. Tim Hardaway Jr led the team as the starting shooting guard, going 6 - for - 13 from the field and 3 - for - 5 from the three - point line to score 17 points, while also adding four assists. He's now scored 17 or more points in three out of his last four games, as he has put it on himself to pick up the slack with other key players sitting out. J.R. Smith also put together a solid outing as a starter. He finished with 15 points and seven rebounds in 37 minutes. Like Haradaway Jr, he's also benefitted from other guys sitting out, and has now combined for 37 points over his last two games. While he did n't have his best night defensively, Cole Aldrich scored 12 points (6 - 10 FG) and grabbed seven rebounds in 19 minutes. The only other Knick to reach double figures in points was Jason Smith, who came off the bench for 10 points (3 - 11 FG, 4 - 4 FT). The Bucks' next game will be at home against the Phoenix Suns on Tuesday, while the Knicks will travel to Memphis to play the Grizzlies on Monday."],
    "Answer": [4]
}
user_instruction = "What is the number of assists for {player_name} in the game {Report} if they are mentioned. Return only the number (only an integer) of assists or -1 if there are no mentions for assists for {player_name} (without explanation)."
start = time()
df_assists = df_merged.sem_map(user_instruction)
end = time()
elapsed_times.append(end-start)
print("Assists Extraction, Elapsed Time: ", end-start, '\n')
df = df_assists.rename(columns={"_map": "assists"})

# Extract blocks
print("Extracting Blocks")
examples = {
    "player_name": ["Jonas Valnciunas"],
    "Report": ["The Toronto Raptors (29 - 15) defeated the Detroit Pistons (17 - 28) 114 - 110 on Sunday at the Air Canada Center in Toronto. Despite being out - scored 31 - 25 in the final quarter, the Raptors were able to hold of the Pistons' late comeback attempt and secure the four - point victory in front of their home crowd. While the game may have been close, the Raptors shot the ball much better than the Pistons, going 53 percent from the field compared to just 46 percent from the field for the Pistons. The Raptors also forced them into 14 turnovers, while only committing eight of their own, which may have made a big difference in this one. After combining for only 14 points over his last three games, DeMar DeRozan returned to form Sunday, finishing with 25 points (8 - 14 FG, 1 - 2 3Pt, 8 - 10 FT), six rebounds and four assists. It was good to see him turn things around, as the Raptors really needed him to play well after losing six of their last ten games. Jonas Valnciunas was another big factor in the win. He went 9 - for - 15 from the field and 2 - for - 2 from the free throw line to score 20 points, while adding 11 rebounds and three blocked shots as well. He's now recorded a double - double in three out of his last four games, while also notching three blocks in two consecutive outings. Amir Johnson had a solid showing as well, finishing with an efficient 17 points (7 - 9 FG, 3 - 4 FT) and two rebounds. He only played 17 minutes in Friday's win over the Sixers, but he was back to a normal amount of minutes Sunday, playing a full game's worth of 28. Both Greivis Vasquez and Louis Williams reached double figures in points as well, with 13 and 12 points respectively. With Brandon Jennings going down for the year, D.J. Augustin stepped up in a big way, going 12 - for - 20 from the field and 5 - for - 9 from the three - point line to score a game - high of 35 points, while adding eight assists as well. It was the most shots he's taken all season, resulting in a new season - high in points. He'll run as the starting point guard moving forward. Greg Monroe had another strong stat line Sunday, recording 21 points (9 - 17 FG, 3 - 5 FT) and 16 rebounds. He's now posted a double - double in four out of his last five games. Andre Drummond nearly notched a double - double of his own, but came up just shy with 14 points (7 - 11 FG, 0 - 1 FT) and eight rebounds. He had a really tough matchup with Valanciunas, so he did n't have his normal eye - popping amount of rebounds that he's come accustomed to. The only other Piston to reach double figures in points was Kentavious Caldwell-Pope who added 16 points (6 - 17 FG, 3 - 9 3Pt, 1 - 2 FT), three rebounds and two steals. The Raptors' next game will be on the road against the Indiana Pacers on Tuesday, while the Pistons will be at home against the Cleveland Cavaliers on Tuesday."],
    "Answer": [3]
}
user_instruction = "What is the number of blocks for {player_name} in the game {Report} if they are mentioned. Return only the number (only an integer) of blocks or -1 if there are no mentions for blocks for {player_name} (without explanation)."
start = time()
df_blocks = df.sem_map(user_instruction)
end = time()
elapsed_times.append(end-start)
print("Blocks Extraction, Elapsed Time: ", end-start, '\n')
df = df_blocks.rename(columns={"_map": "blocks"})

# Extract points
print("Extracting Points")
examples = {
    "player_name": ["Giannis Antetokounmpo"],
    "Report": ["The Milwaukee Bucks (18 - 17) defeated the New York Knicks (5 - 31) 95 - 82 on Sunday at Madison Square Garden in New York. The Bucks were able to have a great night defensively, giving themselves the scoring advantage in all four quarters. The Bucks showed superior shooting, going 46 percent from the field, while the Knicks went only 41 percent from the floor. The Bucks also out - rebounded the Knicks 48 - 36, giving them in an even further advantage which helped them secure the 13 - point victory on the road. Brandon Knight led the Bucks again in this one. He went 6 - for - 14 from the field and 1 - for - 3 from beyond the arc to score 17 points, while also handing out five assists. He's now averaging 21 points per game over his last three games, as he's consistently been the offensive leader for this team. Zaza Pachulia also had a strong showing, finishing with 16 points (6 - 12 FG, 4 - 4 FT) and a team - high of 14 rebounds. It marked his second double - double in a row and fourth on the season, as the inexperienced centers on the Knicks' roster were n't able to limit him. Notching a double - double of his own, Giannis Antetokounmpo recorded 16 points (6 - 9 FG, 1 - 1 3Pt, 3 - 6 FT) and 12 rebounds. The 12 rebounds matched a season - high, while it was his second double - double of the season. Coming off the bench for a big night was Kendall Marshall. He went 6 - for - 8 from the field and 3 - for - 3 from the free throw line to score 15 points in 20 minutes. The Knicks really struggled to score without Carmelo Anthony and Amare Stoudemire. Tim Hardaway Jr led the team as the starting shooting guard, going 6 - for - 13 from the field and 3 - for - 5 from the three - point line to score 17 points, while also adding four assists. He's now scored 17 or more points in three out of his last four games, as he has put it on himself to pick up the slack with other key players sitting out. J.R. Smith also put together a solid outing as a starter. He finished with 15 points and seven rebounds in 37 minutes. Like Haradaway Jr, he's also benefitted from other guys sitting out, and has now combined for 37 points over his last two games. While he did n't have his best night defensively, Cole Aldrich scored 12 points (6 - 10 FG) and grabbed seven rebounds in 19 minutes. The only other Knick to reach double figures in points was Jason Smith, who came off the bench for 10 points (3 - 11 FG, 4 - 4 FT). The Bucks' next game will be at home against the Phoenix Suns on Tuesday, while the Knicks will travel to Memphis to play the Grizzlies on Monday."],
    "Answer": [16]
}
user_instruction = "What is the number of points for {player_name} in the game {Report} if they are mentioned. Return only the number (only an integer) of points or -1 if there are no mentions for points for {player_name} (without explanation)."
start = time()
df_points = df.sem_map(user_instruction)
end = time()
elapsed_times.append(end-start)
print("Points Extraction, Elapsed Time: ", end-start, '\n')
df = df_points.rename(columns={"_map": "points"})

# Extract total rebounds
print("Extracting Total Rebounds")
examples = {
    "player_name": ["Zaza Pachulia"],
    "Report": ["The Milwaukee Bucks (18 - 17) defeated the New York Knicks (5 - 31) 95 - 82 on Sunday at Madison Square Garden in New York. The Bucks were able to have a great night defensively, giving themselves the scoring advantage in all four quarters. The Bucks showed superior shooting, going 46 percent from the field, while the Knicks went only 41 percent from the floor. The Bucks also out - rebounded the Knicks 48 - 36, giving them in an even further advantage which helped them secure the 13 - point victory on the road. Brandon Knight led the Bucks again in this one. He went 6 - for - 14 from the field and 1 - for - 3 from beyond the arc to score 17 points, while also handing out five assists. He's now averaging 21 points per game over his last three games, as he's consistently been the offensive leader for this team. Zaza Pachulia also had a strong showing, finishing with 16 points (6 - 12 FG, 4 - 4 FT) and a team - high of 14 rebounds. It marked his second double - double in a row and fourth on the season, as the inexperienced centers on the Knicks' roster were n't able to limit him. Notching a double - double of his own, Giannis Antetokounmpo recorded 16 points (6 - 9 FG, 1 - 1 3Pt, 3 - 6 FT) and 12 rebounds. The 12 rebounds matched a season - high, while it was his second double - double of the season. Coming off the bench for a big night was Kendall Marshall. He went 6 - for - 8 from the field and 3 - for - 3 from the free throw line to score 15 points in 20 minutes. The Knicks really struggled to score without Carmelo Anthony and Amare Stoudemire. Tim Hardaway Jr led the team as the starting shooting guard, going 6 - for - 13 from the field and 3 - for - 5 from the three - point line to score 17 points, while also adding four assists. He's now scored 17 or more points in three out of his last four games, as he has put it on himself to pick up the slack with other key players sitting out. J.R. Smith also put together a solid outing as a starter. He finished with 15 points and seven rebounds in 37 minutes. Like Haradaway Jr, he's also benefitted from other guys sitting out, and has now combined for 37 points over his last two games. While he did n't have his best night defensively, Cole Aldrich scored 12 points (6 - 10 FG) and grabbed seven rebounds in 19 minutes. The only other Knick to reach double figures in points was Jason Smith, who came off the bench for 10 points (3 - 11 FG, 4 - 4 FT). The Bucks' next game will be at home against the Phoenix Suns on Tuesday, while the Knicks will travel to Memphis to play the Grizzlies on Monday."],
    "Answer": [14]
}
user_instruction = "What is the number of total rebounds for {player_name} in the game {Report} if they are mentioned. Return only the number (only an integer) of total rebounds or -1 if there are no mentions for total rebounds for {player_name} (without explanation)."
start = time()
df_rebounds = df.sem_map(user_instruction)
end = time()
elapsed_times.append(end-start)
print("Total Rebounds Extraction, Elapsed Time: ", end-start, '\n')
df = df_rebounds.rename(columns={"_map": "total_rebounds"})

# Extract steals
print("Extracting Steals")
examples = {
    "player_name": ["Kentavious Caldwell-Pope"],
    "Report": ["The Toronto Raptors (29 - 15) defeated the Detroit Pistons (17 - 28) 114 - 110 on Sunday at the Air Canada Center in Toronto. Despite being out - scored 31 - 25 in the final quarter, the Raptors were able to hold of the Pistons' late comeback attempt and secure the four - point victory in front of their home crowd. While the game may have been close, the Raptors shot the ball much better than the Pistons, going 53 percent from the field compared to just 46 percent from the field for the Pistons. The Raptors also forced them into 14 turnovers, while only committing eight of their own, which may have made a big difference in this one. After combining for only 14 points over his last three games, DeMar DeRozan returned to form Sunday, finishing with 25 points (8 - 14 FG, 1 - 2 3Pt, 8 - 10 FT), six rebounds and four assists. It was good to see him turn things around, as the Raptors really needed him to play well after losing six of their last ten games. Jonas Valnciunas was another big factor in the win. He went 9 - for - 15 from the field and 2 - for - 2 from the free throw line to score 20 points, while adding 11 rebounds and three blocked shots as well. He's now recorded a double - double in three out of his last four games, while also notching three blocks in two consecutive outings. Amir Johnson had a solid showing as well, finishing with an efficient 17 points (7 - 9 FG, 3 - 4 FT) and two rebounds. He only played 17 minutes in Friday's win over the Sixers, but he was back to a normal amount of minutes Sunday, playing a full game's worth of 28. Both Greivis Vasquez and Louis Williams reached double figures in points as well, with 13 and 12 points respectively. With Brandon Jennings going down for the year, D.J. Augustin stepped up in a big way, going 12 - for - 20 from the field and 5 - for - 9 from the three - point line to score a game - high of 35 points, while adding eight assists as well. It was the most shots he's taken all season, resulting in a new season - high in points. He'll run as the starting point guard moving forward. Greg Monroe had another strong stat line Sunday, recording 21 points (9 - 17 FG, 3 - 5 FT) and 16 rebounds. He's now posted a double - double in four out of his last five games. Andre Drummond nearly notched a double - double of his own, but came up just shy with 14 points (7 - 11 FG, 0 - 1 FT) and eight rebounds. He had a really tough matchup with Valanciunas, so he did n't have his normal eye - popping amount of rebounds that he's come accustomed to. The only other Piston to reach double figures in points was Kentavious Caldwell-Pope who added 16 points (6 - 17 FG, 3 - 9 3Pt, 1 - 2 FT), three rebounds and two steals. The Raptors' next game will be on the road against the Indiana Pacers on Tuesday, while the Pistons will be at home against the Cleveland Cavaliers on Tuesday."],
    "Answer": [2]
}
user_instruction = "What is the number of steals for {player_name} in the game {Report} if they are mentioned. Return only the number (only an integer) of steals or -1 if there are no mentions for steals for {player_name} (without explanation)."
start = time()
df_steals = df.sem_map(user_instruction)
end = time()
elapsed_times.append(end-start)
print("Steals Extraction, Elapsed Time: ", end-start, '\n')
df = df_steals.rename(columns={"_map": "steals"})

# # Extract 3-pointers made
# print("Extracting 3-Pointers Made")
# user_instruction = "What is the number of 3-pointers made for {player_name} in the game {Report} if they are mentioned. Return only the number (only an integer) of 3-pointers made or zero if there are no mentions for 3-pointers made for {player_name}."
# start = time()
# df_3_pointers = df.sem_map(user_instruction)
# end = time()
# elapsed_times.append(end-start)
# print("3-Pointers Made Extraction, Elapsed Time: ", end-start, '\n')
# df = df_3_pointers.rename(columns={"_map": "3_pointers_made"})

# # Extract field goals attempted
# print("Extracting Field Goals Attempted")
# user_instruction = "What is the number of field goals attempted for {player_name} in the game {Report} if they are mentioned. Return only the number (only an integer) of field goals attempted or zero if there are no mentions for field goals attempted for {player_name}."
# start = time()
# df_field_goals_attempted = df.sem_map(user_instruction)
# end = time()
# elapsed_times.append(end-start)
# print("Field Goals Attempted Extraction, Elapsed Time: ", end-start, '\n')
# df = df_field_goals_attempted.rename(columns={"_map": "field_goals_attempted"})

# # Extract minutes played
# print("Extracting Minutes Played")
# user_instruction = "What is the number of minutes played for {player_name} in the game {Report} if they are mentioned. Return only the number of minutes played or zero if there are no mentions for minutes played for {player_name}."
# start = time()
# df_minutes_played = df.sem_map(user_instruction)
# end = time()
# elapsed_times.append(end-start)
# print("Minutes Played Extraction, Elapsed Time: ", end-start, '\n')
# df = df_minutes_played.rename(columns={"_map": "minutes_played"})

# # Extract field goals made
# print("Extracting Field Goals Made")
# user_instruction = "What is the number of field goals made for {player_name} in the game {Report} if they are mentioned. Return only the number of field goals made or zero if there are no mentions for field goals made for {player_name}."
# start = time()
# df_field_goals_made = df.sem_map(user_instruction)
# end = time()
# elapsed_times.append(end-start)
# print("Field Goals Made Extraction, Elapsed Time: ", end-start, '\n')
# df = df_field_goals_made.rename(columns={"_map": "field_goals_made"})

# # Extract free throws attempted
# print("Extracting Free Throw Attempted")
# user_instruction = "What is the number of free throws attempted for {player_name} in the game {Report} if they are mentioned. Return only the number of free throws attempted or zero if there are no mentions for free throws attempted for {player_name}."
# start = time()
# df_free_throws_attempted = df.sem_map(user_instruction)
# end = time()
# elapsed_times.append(end-start)
# print("Free Throws Attempted Extraction, Elapsed Time: ", end-start, '\n')
# df = df_free_throws_attempted.rename(columns={"_map": "free_throws_attempted"})

# # Extract free throws made
# print("Extracting Free Throws Made")
# user_instruction = "What is the number of free throws made for {player_name} in the game {Report} if they are mentioned. Return only the number of free throws made or zero if there are no mentions for free throws made for {player_name}."
# start = time()
# df_free_throws_made = df.sem_map(user_instruction)
# end = time()
# elapsed_times.append(end-start)
# print("Free Throws Made Extraction, Elapsed Time: ", end-start, '\n')
# df = df_free_throws_made.rename(columns={"_map": "free_throws_made"})

# # Extract defensive rebounds
# print("Extracting Defensive Rebounds")
# user_instruction = "What is the number of defensive rebounds for {player_name} in the game {Report} if they are mentioned. Return only the number of defensive rebounds or zero if there are no mentions for defensive rebounds for {player_name}."
# start = time()
# df_def_rebounds = df.sem_map(user_instruction)
# end = time()
# elapsed_times.append(end-start)
# print("Defensive Rebounds Extraction, Elapsed Time: ", end-start, '\n')
# df = df_def_rebounds.rename(columns={"_map": "defensive_rebounds"})


# # Extract 3-pointers attempted
# print("Extracting 3-Pointers Attempted")
# user_instruction = "What is the number of 3-pointers attempted for {player_name} in the game {Report} if they are mentioned. Return only the number of 3-pointers attempted or zero if there are no mentions for 3-pointers attempted for {player_name}."
# start = time()
# df_3_pointers_attempted = df.sem_map(user_instruction)
# end = time()
# elapsed_times.append(end-start)
# print("3-Pointers Attempted Extraction, Elapsed Time: ", end-start, '\n')
# df = df_3_pointers_attempted.rename(columns={"_map": "3_pointers_attempted"})

# # Extract personal fouls
# print("Extracting Personal Fouls")
# user_instruction = "What is the number of personal fouls for {player_name} in the game {Report} if they are mentioned. Return only the number of personal fouls or zero if there are no mentions for personal fouls for {player_name}."
# start = time()
# df_personal_fouls = df.sem_map(user_instruction)
# end = time()
# elapsed_times.append(end-start)
# print("Personal Fouls Extraction, Elapsed Time: ", end-start, '\n')
# df = df_personal_fouls.rename(columns={"_map": "personal_fouls"})

# # Extract turnovers
# print("Extracting Turnovers")
# user_instruction = "What is the number of turnovers for {player_name} in the game {Report} if they are mentioned. Return only the number of turnovers or zero if there are no mentions for turnovers for {player_name}."
# start = time()
# df_turnovers = df.sem_map(user_instruction)
# end = time()
# elapsed_times.append(end-start)
# print("Turnovers Extraction, Elapsed Time: ", end-start, '\n')
# df = df_turnovers.rename(columns={"_map": "turnovers"})

# # Extract offensive rebounds
# print("Extracting Offensive Rebounds")
# user_instruction = "What is the number of offensive rebounds for {player_name} in the game {Report} if they are mentioned. Return only the number of offensive rebounds or zero if there are no mentions for offensive rebounds for {player_name}."
# start = time()
# df_off_rebounds = df.sem_map(user_instruction)
# end = time()
# elapsed_times.append(end-start)
# print("Offensive Rebounds Extraction, Elapsed Time: ", end-start, '\n')
# df = df_off_rebounds.rename(columns={"_map": "offensive_rebounds"})

# # Extract field goal percentage
# print("Extracting Field Goal Percentage")
# user_instruction = "What is the field goal percentage for {player_name} in the game {Report} if it is mentioned. Return only the field goal percentage or zero if there is no mention for field goal percentage for {player_name}."
# start = time()
# df_fg_percentage = df.sem_map(user_instruction)
# end = time()
# elapsed_times.append(end-start)
# print("Field Goal Percentage Extraction, Elapsed Time: ", end-start, '\n')
# df = df_fg_percentage.rename(columns={"_map": "field_goal_percentage"})

# # Extract 3-pointer percentage
# print("Extracting 3-Pointer Percentage")
# user_instruction = "What is the 3-pointer percentage for {player_name} in the game {Report} if it is mentioned. Return only the 3-pointer percentage or zero if there is no mention for 3-pointer percentage for {player_name}."
# start = time()
# df_3p_percentage = df.sem_map(user_instruction)
# end = time()
# elapsed_times.append(end-start)
# print("3-Pointer Percentage Extraction, Elapsed Time: ", end-start, '\n')
# df = df_3p_percentage.rename(columns={"_map": "3_pointer_percentage"})

# # Extract free throw percentage
# print("Extracting Free Throw Percentage")
# user_instruction = "What is the free throw percentage for {player_name} in the game {Report} if it is mentioned. Return only the free throw percentage or zero if there is no mention for free throw percentage for {player_name}."
# start = time()
# df_ft_percentage = df.sem_map(user_instruction)
# end = time()
# elapsed_times.append(end-start)
# print("Free Throw Percentage Extraction, Elapsed Time: ", end-start, '\n')
# df = df_ft_percentage.rename(columns={"_map": "free_throw_percentage"})

exec_time = sum(elapsed_times)

if args.wandb:
    if args.provider == 'ollama':
        output_file = f"evaluation/derivation/Q1/results/lotus_Q1_map_{args.model.replace(':', '_')}_{args.provider}_{args.size}.csv"
    elif args.provider =='vllm':
        output_file = f"evaluation/derivation/Q1/results/lotus_Q1_map_{args.model.replace('/', '_')}_{args.provider}_{args.size}.csv"
        
    df.to_csv(output_file)

    wandb.log({
        "result_table": wandb.Table(dataframe=df),
        "execution_time": exec_time
    })

    wandb.finish()
else:
    print("Result:\n\n", df)
    print("Execution time: ", exec_time)


