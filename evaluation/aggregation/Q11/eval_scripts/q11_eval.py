import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--size", nargs='?', default=1000, const=1000, type=int, help="The input size")
args = parser.parse_args()

imdb_reviews = pd.read_csv("datasets/imdb_reviews/imdb_reviews.csv").head(args.size)

imdb_reviews = pd.DataFrame(imdb_reviews['sentiment'].value_counts())

print(imdb_reviews.loc[imdb_reviews['count'].idxmax()])