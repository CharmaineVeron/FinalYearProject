from tracemalloc import start
import pandas as pd
import numpy as np
import openpyxl
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

# Import Telecom Data
df = pd.read_excel (r'C:\Users\user\Documents\FYP\FYPdata.xlsx', sheet_name='sheet1', usecols="B:E",  engine='openpyxl')

# Find data by specific Date, 5 here refers to June 5th
df = df.loc[(df['date'] == 5)]

# Calculate Workload for every minute
df['Workloadmin'] = abs((df['start time'] - df['end time']).astype('timedelta64[m]')).astype(int)
df['start time'] = df['start time'].dt.floor('Min')
df['end time'] = df['end time'].dt.floor('Min')
df['start time'] = pd.to_datetime(df['start time'])
interim = []

# Represent workload 
for i,row in df.iterrows():
    num = int(df['Workloadmin'].loc[i])
    starttime = df['start time'].loc[i]
    a=0
    while a < num:
        interim.append([starttime + timedelta(minutes=a)])
        a+=1

# To panda dataframe
interim = pd.DataFrame(interim, columns=['start time'])
df = pd.concat([df, interim], sort=False)

df = df.sort_values(by="start time")
count = df['start time'].value_counts(sort=False)

df2 = pd.DataFrame({'start time':count.index, 'count':count.values})


# Visualization of workload patterns
X = df2['start time']
Y = df2['count']

plt.plot(X,Y)
plt.ylabel("Requests")
plt.xlabel("Minutes")
plt.grid()
plt.show()

# Write to prepared data seperate file
out_path = "C:\\Users\\user\\Documents\\FYP\\ProcessedData.xlsx"

writer = pd.ExcelWriter(out_path , engine='xlsxwriter', datetime_format='m/d/yyyy hh:mm:ss',)

df2.to_excel(writer)
writer.save()
