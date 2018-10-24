import pandas as pd

thickness_lh_df = pd.read_csv("../data/aparcstat_a2009s_thickness_lh.txt",
                              delimiter="\t")
thickness_rh_df = pd.read_csv("../data/aparcstat_a2009s_thickness_rh.txt",
                              delimiter="\t")
volume_lh_df = pd.read_csv("../data/aparcstat_a2009s_volume_lh.txt",
                           delimiter="\t")
volume_rh_df = pd.read_csv("../data/aparcstat_a2009s_volume_rh.txt",
                           delimiter="\t")

# Each dataframe and their primary key.
dataframes = [thickness_lh_df, thickness_rh_df, volume_lh_df, volume_rh_df]
primary_keys = ["lh.aparc.a2009s.thickness", "rh.aparc.a2009s.thickness",
                "lh.aparc.a2009s.area",
                "rh.aparc.a2009s.area"]

# For each dataframe, split the primary key to get the patient number and
# date of the scan. For the date of the scan, only keep the month and the year,
# to match the dates in the PPMI tables.
for x in range(0, 4):
    df = dataframes[x]
    pk = primary_keys[x]
    date_scan = []
    patient_no = []
    for s in df[pk]:
        details_split = s.split("_")
        patient_info = details_split[1].split("/")
        date = details_split[2]
        date_split = date.split("-")
        date_scan.append(date_split[1] + "/" + date_split[0])
        patient_no.append(patient_info[1])
    df["date_scan"] = date_scan
    df["patno"] = patient_no

data = pd.merge(dataframes[0], dataframes[1], on=["date_scan", "patno"])
data = pd.merge(data, dataframes[2], on=["date_scan", "patno"])
data = pd.merge(data, dataframes[3], on=["date_scan", "patno"])

data = data.drop(columns=primary_keys)

data.to_csv("../data/thickness_and_volume_data.csv")
