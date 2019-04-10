import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile, chi2, f_classif

updrs_df = pd.read_csv("../data/MDS_UPDRS_Part_III.csv")
thick_vol_df = pd.read_csv("../data/thickness_and_volume_data.csv")

# The fields in the UPDRS file that denote scores.
scoring_fields = ["NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU", "NP3RIGLU",
                  "PN3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR",
                  "NP3HMOVL", "NP3PRSPR", "NP3PRSPL", "NP3TTAPR", "NP3TTAPL",
                  "NP3LGAGR", "NP3LGAGL", "NP3RISNG", "NP3GAIT", "NP3FRZGT",
                  "NP3PSTBL", "NP3POSTR", "NP3BRADY", "NP3PTRMR", "NP3PTRML",
                  "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL",
                  "NP3RTALL", "NP3RTALJ", "NP3RTCON"]

# Compute the UPDRS scores by summing up all categories, replacing NaNs with 0.
updrs_df["score"] = 0
for field in scoring_fields:
    updrs_df[field].fillna(0, inplace=True)
for field in scoring_fields:
    updrs_df["score"] += updrs_df[field]

updrs_df.rename(columns={"INFODT": "date_scan", "PATNO": "patno"}, inplace=True)
updrs_df['patno'] = updrs_df['patno'].apply(str)
updrs_df = updrs_df[["patno", "score", "date_scan"]]

thick_vol_df['patno'] = thick_vol_df['patno'].apply(str)
thick_vol_df = pd.merge(thick_vol_df, updrs_df, on=["date_scan", "patno"])
thick_vol_df.drop_duplicates(subset=["date_scan", "patno"], keep="first",
                             inplace=True)

# Split the scores into equally-sized buckets.
num_buckets = 4
max_bucket_size = len(thick_vol_df["score"]) / num_buckets
scoring_buckets = {}
bucket = 0
current_bucket_size = 0
counts = thick_vol_df["score"].value_counts().to_dict()
for score in sorted(counts):
    count = counts[score]
    if current_bucket_size > max_bucket_size and score != 0:
        if bucket < num_buckets - 1:
            bucket += 1
        current_bucket_size = 0
    scoring_buckets[score] = bucket
    current_bucket_size += count


def find_most_common_score(scoring_buckets, counts, num_buckets):
    mcs = {x: 0 for x in range(num_buckets)}
    max_count = {x: 0 for x in range(num_buckets)}
    for score, bucket in scoring_buckets.items():
        if counts[score] > max_count[bucket]:
            mcs[bucket] = score
            max_count[bucket] = counts[score]
    return mcs


# print(scoring_buckets)
# print(find_most_common_score(scoring_buckets, counts, num_buckets))
thick_vol_df["score"].replace(scoring_buckets, inplace=True)
print(thick_vol_df["score"].value_counts())


fs = SelectPercentile(f_classif, percentile=100)
best = fs.fit(thick_vol_df.drop(columns=["date_scan", "patno", "score"]), thick_vol_df["score"]).get_support()
cols = thick_vol_df.columns.values
scores = []
max_score = 0
best_col = ""
for col, val, score in zip(cols, best, fs.scores_):
    if val and score > max_score:
        max_score = score
        best_col = col

# print(best_col)
# print(sum(scores) / len(scores))
# print(SelectPercentile(chi2, percentile=10).fit(thick_vol_df.drop(columns=["date_scan", "patno", "score"]), thick_vol_df["score"]).get_support())
# thick_vol_df.to_csv("../data/updrs.csv", index=False)
