import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

data = pd.read_csv("pima-indians-diabetes.csv",skiprows=9,header=None)
file = open("pima-indians-diabetes.csv")
lines = file.readlines()[0:9]

headers = []
for line in lines:
    headers.append(line[5:-1])
data_as_np = numpy.asarray(data)
df = pd.DataFrame(data=data_as_np,columns=headers)
data_as_np = df.to_numpy()

print(df)
print(data_as_np)

input_data = data_as_np[:,0:8]
output_data = data_as_np[:,-1] 

X_train, X_test, y_train, y_test = train_test_split(
    input_data, output_data, test_size=0.2, random_state=True
)

Log_RegressionModel = LogisticRegression()
Log_RegressionModel.fit(X_train,y_train)
y_pred = Log_RegressionModel.predict(X_test)

correctly_classified = X_test[y_test == y_pred]
misclassified = X_test[y_test != y_pred]

conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrica zabune:")
print(conf_matrix)

# Izračun točnosti, preciznosti i odziva
accuracy = accuracy_score(y_test, y_pred)
print("Tocnost:", accuracy)

precision = precision_score(y_test, y_pred)
print("Preciznost:", precision)

recall = recall_score(y_test, y_pred)
print("Odziv:", recall)

f1 = f1_score(y_test,y_pred)
print("F1:", recall)

cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()