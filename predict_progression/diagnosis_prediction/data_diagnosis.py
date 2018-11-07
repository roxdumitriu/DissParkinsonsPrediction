import pandas as pd

diagnosis_df = pd.read_csv("../../data/Patient_Status.csv")
thick_vol_df = pd.read_csv("../../data/thickness_and_volume_data.csv")

diagnosis_df.rename(columns={"RECRUITMENT_CAT": "diagnosis", "PATNO": "patno"},
                    inplace=True)
diagnosis_df = diagnosis_df[["patno", "diagnosis"]]
diagnosis_df['patno'] = diagnosis_df['patno'].apply(str)

thick_vol_df['patno'] = thick_vol_df['patno'].apply(str)
thick_vol_df = pd.merge(thick_vol_df, diagnosis_df, on=["patno"])

print(thick_vol_df["diagnosis"].value_counts())

thick_vol_df.to_csv("../../data/diagnosis.csv", index=False)

