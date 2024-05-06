import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("pima-indians-diabetes.csv",skiprows=9,header=None)
file = open("pima-indians-diabetes.csv")
lines = file.readlines()[0:9]

headers = []
for line in lines:
    headers.append(line[5:-1])
data_as_np = numpy.asarray(data)
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

model=keras.Sequential()
model.add(layers.Input((8,)))
# model.add (layers.Flatten())
model.add(layers.Dense(12,activation="relu"))
model.add(layers.Dense(8,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))
model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
batch_size = 10
epochs = 150
history = model.fit(X_train_n, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1) 
score = model.evaluate(X_test_n, y_test, verbose=0)
print("Test accuracy:", score[1])

model.save("zad3_model.keras")
