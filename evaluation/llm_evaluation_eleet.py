import pandas as pd
from litellm import completion

df_labels = pd.read_csv('players_labels_100.csv')
df_eleet = pd.read_csv('derivation/NER_TE/Q1/eleet.csv')

def llm_match_name(name, choices):
    if not choices:
        return None

    prompt = f"""
    You are matching basketball player names.
    Target name: "{name}"
    Candidate names: {choices}

    If one of the candidates clearly refers to the same player, return that exact candidate string.
    If no good match, return None.
    """

    resp = completion(
        model="ollama/gemma3:1b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20
    )

    result = resp["choices"][0]["message"]["content"].strip()
    return None if result.lower() == "none" else result

def match_group(group):
    game_id = group.name
    choices = df_labels.loc[df_labels['Game ID'] == game_id, 'Player Name'].tolist()
    group['matched_player'] = group['name'].apply(lambda x: llm_match_name(x, choices))
    return group

df_eleet = df_eleet.groupby('Game ID', group_keys=False).apply(match_group)

df_eleet.to_csv("testest.csv")

df = df_labels.merge(
    df_eleet,
    left_on=['Game ID', 'Player Name'],
    right_on=['Game ID', 'matched_player'],
    how='left',
    indicator=True
)

str_to_num = {
    "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "eleven": 11.0, "twelve": 12.0, "thirteen": 13.0, "fourteen": 14.0,
    "fifteen": 15.0, "sixteen": 16.0, "seventeen": 17.0, "eighteen": 18.0,
    "nineteen": 19.0, "twenty": 20.0,
}
cols_to_replace = ["Points_y", "Assists_y", "Total rebounds_y", "Blocks_y", "Steals_y"]

df[cols_to_replace] = df[cols_to_replace].replace(str_to_num)
df[cols_to_replace] = df[cols_to_replace].apply(pd.to_numeric, errors='coerce')

df.drop(
    columns=[
        "Defensive rebounds", "Offensive rebounds", "3-pointers attempted", "3-pointers made",
        "Field goals attempted", "Field goals made", "Free throws attempted", "Free throws made",
        "Minutes played", "Personal fouls", "Turnovers", "Field goal percentage",
        "Free throw percentage", "3-pointer percentage"
    ],
    inplace=True
)

cols = ["Points", "Assists", "Total rebounds", "Blocks", "Steals"]

for col in cols:
    xcol, ycol = f"{col}_x", f"{col}_y"
    df[f"{col}_match"] = (df[xcol].fillna(-1) == df[ycol].fillna(-1))

for col in cols:
    acc = df[f"{col}_match"].mean()
    print(f"{col} accuracy: {acc:.2%}")

total_accuracy = df[[f"{col}_match" for col in cols]].stack().mean()
print(f"Total accuracy: {total_accuracy:.2%}")
