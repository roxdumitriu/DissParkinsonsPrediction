import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../data/updrs.csv")
cols = data.columns.values
cols = sorted(cols)
for col in data.columns.values:
    xs = []
    for x in data['score'].unique():
        xs.append(list(data[data['score'] == x][col]))

    # Assign colors for each airline and the names
    # colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
    names = ['0', '1']

    # Make the histogram using a list of lists
    # Normalize the flights and assign colors and names
    plt.hist(xs, bins=int(180 / 15), label=data['score'].unique())

    # Plot formatting
    plt.legend()
    plt.xlabel(col)
    plt.ylabel('Score Buckets')
    plt.title('Side-by-Side Histogram with Multiple Score Buckets')
    plt.show()
