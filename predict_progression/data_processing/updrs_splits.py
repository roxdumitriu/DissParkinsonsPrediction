import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

N_SPLITS = 10

updrs_df = pd.read_csv("../../data/updrs.csv")
max_score = int(pd.Series.max(updrs_df["score"])) + 1
splits = [pd.DataFrame() for x in range(0, N_SPLITS)]

for score in range(0, max_score):
    s = updrs_df.loc[updrs_df["score"] == score]
    split_size = int(s.shape[0] / N_SPLITS)
    x = 0
    for i in range(0, len(splits)):
        splits[i] = pd.concat([splits[i], s.iloc[x: x + split_size]])
        x += split_size

for x in range(0, len(splits)):
    print(splits[x]["score"].value_counts())
    splits[x].to_csv("../../data/updrs_splits/split_{}.csv".format(x), index=False)

