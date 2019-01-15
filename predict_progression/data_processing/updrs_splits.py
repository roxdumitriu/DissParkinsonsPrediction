import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

N_SPLITS = 10

updrs_df = pd.read_csv("../../data/updrs.csv")
max_score = int(pd.Series.max(updrs_df["score"])) + 1
splits = []

X = updrs_df.drop(columns=["score"])
y = updrs_df["score"].astype(int)

skf = StratifiedKFold(n_splits=10)
for train_index, test_index in skf.split(X, y):
    X_split = X.iloc[test_index]
    y_split = y.iloc[test_index]
    split = pd.DataFrame(X_split)
    split["score"] = y_split.tolist()
    splits.append(split)

for x in range(0, len(splits)):
    print(splits[x]["score"].value_counts())
    splits[x].to_csv("../../data/updrs_splits/split_{}.csv".format(x),
                     index=False)
