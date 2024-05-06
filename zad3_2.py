import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from tensorflow import keras
from keras import layers, models
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("pima-indians-diabetes.csv",skiprows=9,header=None)
file = open("pima-indians-diabetes.csv")
lines = file.readlines()[0:9]

headers = []
for line in lines:
    headers.append(line[5:-1])
data_as_np = np.asarray(data)
df = pd.DataFrame(data=data_as_np,columns=headers)
data_as_np = df.to_numpy()

input_data = data_as_np[:,0:8]

output_data = data_as_np[:,-1]

X_train, X_test, y_train, y_test = train_test_split(
    input_data, output_data, test_size=0.2, random_state=4
)

scaler = StandardScaler()
X_train_n = scaler.fit_transform(X_train)
X_test_n = scaler.transform(X_test)


model = models.load_model("zad3_model.keras")

predictions = model.predict(X_test_n)
print(predictions)
print(y_test)
rounded_predictions = np.where(predictions > 0.5, 1., 0.)

print(rounded_predictions[:,0])
y_pred = np.argmax(predictions, axis=0)
conf_matrix = confusion_matrix(y_test, rounded_predictions)
cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()