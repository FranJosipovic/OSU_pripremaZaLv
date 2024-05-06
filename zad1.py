import pandas as pd
import numpy
import matplotlib.pyplot as plt

data = pd.read_csv("pima-indians-diabetes.csv",skiprows=9,header=None)
file = open("pima-indians-diabetes.csv")
lines = file.readlines()[0:9]

headers = []
for line in lines:
    headers.append(line[5:-1])

data_as_np = numpy.asarray(data)

df = pd.DataFrame(data=data_as_np,columns=headers)

#1
print(data_as_np[:,0].size) #-768

df.drop_duplicates(subset=['Body mass index (weight in kg/(height in m)^2)','Age (years)'],inplace=True)
    
data_as_np = df.to_numpy()
print(data_as_np[:,0].size)

df.dropna(subset=['Body mass index (weight in kg/(height in m)^2)','Age (years)'],inplace=True)
data_as_np = df.to_numpy()
print(data_as_np[:,0].size)

df = df[(df['Body mass index (weight in kg/(height in m)^2)'] > 0) & (df['Age (years)'] > 0)]
data_as_np = df.to_numpy()
print(data_as_np[:,0].size)

df.plot.scatter(x='Age (years)',y='Body mass index (weight in kg/(height in m)^2)')
plt.title("Odnos dobi i BMI")
plt.xlabel("Godine")
plt.ylabel("BMI kg/(m)^2")
plt.show()

bmi = data_as_np[:,5]
print(max(bmi))
print(min(bmi))
print(bmi.mean())

with_diabetes = df[df['Class variable (0 or 1)'] == 1]
data_as_np_with_diabees = with_diabetes.to_numpy()
print(data_as_np_with_diabees[:,0].size)
bmi_with_diabetes = data_as_np_with_diabees[:,5]
print(max(bmi_with_diabetes))
print(min(bmi_with_diabetes))
print(bmi_with_diabetes.mean())

without_diabetes = df[df['Class variable (0 or 1)'] == 0]
data_as_np_without_diabees = without_diabetes.to_numpy()
print(data_as_np_without_diabees[:,0].size)
bmi_without_diabetes = data_as_np_without_diabees[:,5]
print(max(bmi_without_diabetes))
print(min(bmi_without_diabetes))
print(bmi_without_diabetes.mean())