import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv("../data/updrs.csv")
cols = data.columns.values
cols = sorted(cols)
for col1 in data.columns.values:
    for col2 in data.columns.values:
        if col1 != col2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(data[col1], data[col2], c=data["score"], lw=0)
            plt.show()