import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", nargs='?', default='gemma3:12b', const='gemma3:12b', type=str, help="The model to use")
parser.add_argument("-p", "--provider", nargs='?', default='ollama', const='ollama', type=str, help="The provider of the model")
args = parser.parse_args()

def count_true_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['founded_gt'] < 1970)])

def count_false_positives(df):
    return len(df[(df['_merge'] == 'both') & (df['founded_gt'] >= 1970)])

def count_true_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['founded_gt'] >= 1970)])

def count_false_negatives(df):
    return len(df[(df['_merge'] == 'left_only') & (df['founded_gt'] < 1970)])

if __name__ == "__main__":
    if args.provider == 'ollama':
        results_file = f"evaluation/selection/Q8/results/lotus_Q8_filter_default_{args.model.replace(':', '_')}_{args.provider}.csv"
    elif args.provider == 'vllm':
        results_file = f"evaluation/selection/Q8/results/lotus_Q8_filter_default_{args.model.replace('/', '_')}_{args.provider}.csv"

    team_evidence = pd.read_csv("datasets/rotowire/team_evidence.csv")
    team_evidence = team_evidence[['Team Name', 'founded']]

    lotus_res_default = pd.read_csv(results_file)
    lotus_res_default = lotus_res_default.rename(columns={'team_name' : "Team Name"})
    lotus_res_default['founded'] = 1900 # Just a random value < 1970

    df = team_evidence.merge(lotus_res_default, on=['Team Name'], how='outer', suffixes=('_gt', '_pred'), indicator=True)

    tp = count_true_positives(df)
    fp = count_false_positives(df)
    tn = count_true_negatives(df)
    fn = count_false_negatives(df)

    print("--- Default Implementation ---")
    print(f"True Positives: {tp}"
        f"\nFalse Positives: {fp}"
        f"\nTrue Negatives: {tn}"
        f"\nFalse Negatives: {fn}")

    print(f"Accuracy for default implementation: {(tp + tn) / (tp + tn + fp + fn):.2f}")

    if args.provider == 'ollama':
        results_file = f"evaluation/selection/Q8/results/lotus_Q8_filter_cascades_{args.model.replace(':', '_')}_{args.provider}.csv"
    elif args.provider == 'vllm':
        exit(0)
        # results_file = f"evaluation/selection/Q8/results/lotus_Q8_filter_default_{args.model.replace('/', '_')}_{args.provider}.csv"

    lotus_res_opt = pd.read_csv(results_file)
    lotus_res_opt = lotus_res_opt.rename(columns={'team_name' : "Team Name"})
    lotus_res_opt['founded'] = 1900 # Just a random value < 1970

    df = team_evidence.merge(lotus_res_opt, on=['Team Name'], how='outer', suffixes=('_gt', '_pred'), indicator=True)

    tp = count_true_positives(df)
    fp = count_false_positives(df)
    tn = count_true_negatives(df)
    fn = count_false_negatives(df)

    print("--- Optimized Implementation ---")
    print(f"True Positives: {tp}"
        f"\nFalse Positives: {fp}"
        f"\nTrue Negatives: {tn}"
        f"\nFalse Negatives: {fn}")

    print(f"Accuracy for optimized implementation: {(tp + tn) / (tp + tn + fp + fn):.2f}")
