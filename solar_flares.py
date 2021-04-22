import pandas as pd
import numpy as np

data1 = pd.read_csv('flare.data1', sep = r"\s*", skiprows = [0], header = None, engine = "python")
data2 = pd.read_csv('flare.data2', sep = r"\s*", skiprows = [0], header = None, engine = "python")

print(data1.head())
print(data2.head())
print(data1.iloc[1,1])
print(data1.shape)

# integer encoding
classes = {'A':0, 'B':1, 'C':2, 'D':4, 'E':5, 'F':6, 'H':6}
spotSize = {'X':0, 'R':1, 'S':2, 'A:':3, 'H':4, 'K':5}
spotDistribution = {'X':0, 'O':1, 'I':2, 'C':3}

data = np.empty(data1.shape)

for i in range(data1.shape[0]):
    data[i,0] = classes[data1.iloc[i,0]]
    print(data[i,0])


