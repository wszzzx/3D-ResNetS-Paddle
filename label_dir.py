import pandas as pd
import numpy as np

label_csv_path = 'work/split_ucf/classInd.txt'
data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
labels = {}
for i in range(data.shape[0]):
    label = data.iloc[i, 1]
    number = data.iloc[i, 0]
    labels[label] = number
np.save('label_dir.npy', labels)
print(labels)