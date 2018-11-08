import pandas as pd

moca_df = pd.read_csv("../data/MoCA_.csv")
thick_vol_df = pd.read_csv("../data/thickness_and_volume_data.csv")

moca_df.rename(columns={"INFODT": "date_scan", "PATNO": "patno", "MCATOT":"score"}, inplace=True)
moca_df['patno'] = moca_df['patno'].apply(str)
moca_df = moca_df[["patno", "score", "date_scan"]]

for index, row in moca_df.iterrows():
    year_scan = row["date_scan"].split("/")[1]
    moca_df.at[index, "date_scan"] = year_scan

for index, row in thick_vol_df.iterrows():
    year_scan = row["date_scan"].split("/")[1]
    thick_vol_df.at[index, "date_scan"] = year_scan

thick_vol_df['patno'] = thick_vol_df['patno'].apply(str)
thick_vol_df = pd.merge(thick_vol_df, moca_df, on=["date_scan", "patno"])
thick_vol_df.drop_duplicates(subset=["date_scan", "patno"], keep="first",
                             inplace=True)

# Split the scores into equally-sized buckets.
num_buckets = 3
max_bucket_size = len(thick_vol_df["score"]) / num_buckets
scoring_buckets = {}
bucket = 0
current_bucket_size = 0
counts = thick_vol_df["score"].value_counts().to_dict()
for score in sorted(counts):
    count = counts[score]
    if current_bucket_size + count > max_bucket_size:
        if bucket < num_buckets - 1:
            bucket += 1
        current_bucket_size = 0
    scoring_buckets[score] = bucket
    current_bucket_size += count
print(scoring_buckets)
thick_vol_df["score"].replace(scoring_buckets, inplace=True)
thick_vol_df.to_csv("../data/moca.csv", index=False)
