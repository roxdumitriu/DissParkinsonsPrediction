import pandas as pd

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
scoring_buckets = {}
bucket = 0
for x in range(1, 109):
    if 0 <= x <= 10:
        scoring_buckets[x] = 0
    elif 11 <= x <= 20:
        scoring_buckets[x] = 1
    elif 21 <= x <= 30:
        scoring_buckets[x] = 2
    elif 31 <= x <= 40:
        scoring_buckets[x] = 3
    else:
        scoring_buckets[x] = 4

updrs_df["score"] = 0
for field in scoring_fields:
    updrs_df[field].fillna(0, inplace=True)
for field in scoring_fields:
    updrs_df["score"] += updrs_df[field]
updrs_df["score"].replace(scoring_buckets, inplace=True)

updrs_df.rename(columns={"INFODT": "date_scan", "PATNO": "patno"}, inplace=True)
updrs_df['patno'] = updrs_df['patno'].apply(str)
updrs_df = updrs_df[["patno", "score", "date_scan"]]

thick_vol_df['patno'] = thick_vol_df['patno'].apply(str)
thick_vol_df = pd.merge(thick_vol_df, updrs_df, on=["date_scan", "patno"])
thick_vol_df.drop_duplicates(subset=["date_scan", "patno"], keep="first",
                             inplace=True)

thick_vol_df.to_csv("../data/updrs.csv")
