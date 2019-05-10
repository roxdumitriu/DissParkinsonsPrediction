import pandas as pd

BRAIN_DATA_PATHS = ["../data/aparcstat_a2009s_thickness_lh.txt",
                    "../data/aparcstat_a2009s_thickness_rh.txt",
                    "../data/aparcstat_a2009s_volume_lh.txt",
                    "../data/aparcstat_a2009s_volume_rh.txt"]

PRIMARY_KEYS = ["lh.aparc.a2009s.thickness", "rh.aparc.a2009s.thickness",
                "lh.aparc.a2009s.area",
                "rh.aparc.a2009s.area"]

DIAGNOSIS_PATH = "../data/Patient_Status.csv"

PATNO = "patno"
DATE_SCAN = "date_scan"
DIAGNOSIS = "diagnosis"


def process_brain_data(df_path, pk):
    """ Process one brain data file to be concatenated with the rest.
     Parameters
     ----------
     df_path : string
         Path to the brain data file.
     pk : string
         The primary key of the dataframe. Contains the patient number and the
         date of the scans.
     Returns
     ----------
     A processed data frame, with a separate column for patient number and date
     of scan.
     """
    df = pd.read_csv(df_path, delimiter="\t")
    date_scan = []
    patient_no = []
    for s in df[pk]:
        details_split = s.split("_")
        patient_info = details_split[1].split("/")
        date = details_split[2]
        date_split = date.split("-")
        date_scan.append(date_split[1] + "/" + date_split[0])
        patient_no.append(patient_info[1])
    df[DATE_SCAN] = date_scan
    df[PATNO] = patient_no
    df = df.drop(columns=[pk])

    # Drop all the patients that have a cortical thickness or volume of 0.
    df = df[(df != 0).all(1)]

    return df


def inter_cranial_correction(df):
    """ Scales the size of the brain areas to account for variations in head
        size. ICV should only be applied to volumetric brain data.
     Parameters
     ----------
     df : Pandas DataFrame
         DataFrame containing raw volumes of brain ares.
     Returns
     ----------
     A processed DataFrame, with applied ICV.
     """
    eTIVs = {}
    for diagnosis in list(df[DIAGNOSIS].unique()):
        eTIVs[diagnosis] = df.loc[df[DIAGNOSIS] == diagnosis][
            "eTIV"].mean()
    for column in list(df.columns.values):
        if column in ["eTIV", "BrainSegVolNotVent", DATE_SCAN,
                      PATNO, DIAGNOSIS]:
            continue
        for index, row in df.iterrows():
            df.at[index, column] = row[column] * eTIVs[
                row[DIAGNOSIS]] / row["eTIV"]
    return df


def get_diagnosis(diagnosis_path):
    """ Read and return the diagnosis file. Diagnosis is used to filter out
        prodromal patients (noisy data) and to apply ICV correction.
     Parameters
     ----------
     diagnosis_path : string
         Path to the diagnosis files.
     Returns
     ----------
     The diagnosis DataFrame.
     """
    diagnosis_df = pd.read_csv(diagnosis_path)
    diagnosis_df.rename(
        columns={"RECRUITMENT_CAT": DIAGNOSIS, "PATNO": PATNO},
        inplace=True)
    diagnosis_df = diagnosis_df[[PATNO, DIAGNOSIS]]
    diagnosis_df[PATNO] = diagnosis_df[PATNO].apply(str)

    return diagnosis_df


def concatenate_data(dataframe_paths=BRAIN_DATA_PATHS, pks=PRIMARY_KEYS,
                     diagnosis_path=DIAGNOSIS_PATH):
    """ Concatenate the cortical thicknesses and volumes of the brain areas into
        one DataFrame. Volumes are corrected using ICV.
     Parameters
     ----------
     dataframe_paths : list
         List of paths where to find the cortical thicknesses and volumes files.
     pks : list
         List of primary keys in the files, to be able to extract date of scan
         and patient number.
     diagnosis_path : string
         Path to the diagnosis file.
     Returns
     ----------
     Concatenated DataFrame containing all the features in the final dataset.
     """
    diagnosis_df = get_diagnosis(diagnosis_path)
    data = pd.DataFrame()
    for dataframe, pk in zip(dataframe_paths, pks):
        df = process_brain_data(dataframe, pk)
        # Only keep healthy or diagnosed patients.
        df = pd.merge(df, diagnosis_df, on=[PATNO])
        df = df.loc[df[DIAGNOSIS] != "PRODROMA"]
        if "volume" in dataframe:
            df = inter_cranial_correction(df)
        df = df.drop(columns=[DIAGNOSIS, "eTIV", "BrainSegVolNotVent"])
        data = df if data.empty else pd.merge(data, df, on=[DATE_SCAN, PATNO])

    # Drop irrelevant information.
    data = data.drop(
        columns=["lh_MeanThickness_thickness", "rh_MeanThickness_thickness",
                 "lh_WhiteSurfArea_area", "rh_WhiteSurfArea_area"])
    data = pd.merge(data, diagnosis_df, on=[PATNO])
    data[DIAGNOSIS] = pd.get_dummies(data[DIAGNOSIS])

    return data


data = concatenate_data(BRAIN_DATA_PATHS, PRIMARY_KEYS, DIAGNOSIS_PATH)
data.to_csv("../data/thickness_and_volume_data.csv", index=False)
