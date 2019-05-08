import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile, chi2, f_classif

UPDRS_PATH = "../data/MDS_UPDRS_Part_III.csv"
BRAIN_DATA_PATH = "../data/thickness_and_volume_data.csv"

SCORING_FIELDS = ["NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU", "NP3RIGLU",
                  "PN3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR",
                  "NP3HMOVL", "NP3PRSPR", "NP3PRSPL", "NP3TTAPR", "NP3TTAPL",
                  "NP3LGAGR", "NP3LGAGL", "NP3RISNG", "NP3GAIT", "NP3FRZGT",
                  "NP3PSTBL", "NP3POSTR", "NP3BRADY", "NP3PTRMR", "NP3PTRML",
                  "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL",
                  "NP3RTALL", "NP3RTALJ", "NP3RTCON"]

PATNO = "patno"
DATE_SCAN = "date_scan"
SCORE = "score"


def get_updrs_scores(updrs_path, scoring_fields):
    updrs_df = pd.read_csv(updrs_path)
    updrs_df[SCORE] = 0
    for field in scoring_fields:
        # Replace NaNs with 0s.
        updrs_df[field].fillna(0, inplace=True)
    for field in scoring_fields:
        updrs_df[SCORE] += updrs_df[field]

    updrs_df.rename(columns={"INFODT": DATE_SCAN, "PATNO": PATNO},
                    inplace=True)
    updrs_df[PATNO] = updrs_df[PATNO].apply(str)
    updrs_df = updrs_df[[PATNO, SCORE, DATE_SCAN]]
    return updrs_df


def get_brain_data(brain_data_path):
    thick_vol_df = pd.read_csv(brain_data_path)
    thick_vol_df[PATNO] = thick_vol_df[PATNO].apply(str)
    return thick_vol_df


def bucketise_scores(df, num_buckets):
    max_bucket_size = len(df[SCORE]) / num_buckets
    scoring_buckets = {}
    bucket = 0
    current_bucket_size = 0
    counts = df[SCORE].value_counts().to_dict()
    for score in sorted(counts):
        count = counts[score]
        if current_bucket_size > max_bucket_size and score != 0:
            if bucket < num_buckets - 1:
                bucket += 1
            current_bucket_size = 0
        scoring_buckets[score] = bucket
        current_bucket_size += count

    return scoring_buckets


def form_dataset(updrs_path, brain_data_path, scoring_fields):
    updrs_df = get_updrs_scores(updrs_path, scoring_fields)
    brain_data_df = get_brain_data(brain_data_path)

    brain_data_df = pd.merge(brain_data_df, updrs_df, on=[DATE_SCAN, PATNO])
    brain_data_df.drop_duplicates(subset=[DATE_SCAN, PATNO], keep="first",
                                 inplace=True)
    scoring_buckets = bucketise_scores(brain_data_df, num_buckets=4)
    brain_data_df["score"].replace(scoring_buckets, inplace=True)

    return brain_data_df


thick_vol_df = form_dataset(UPDRS_PATH, BRAIN_DATA_PATH, SCORING_FIELDS)
thick_vol_df.to_csv("../data/updrs.csv", index=False)
