import json
import pandas as pd
from matplotlib import pyplot as plt

max_list = []
min_list =[]
average_list = []

with open("plot/records.json") as json_file:
    json_data = json.load(json_file)

df = pd.DataFrame(json_data)
df.set_index("generation")
df[["max", "min", "avg"]].plot()
plt.show()
