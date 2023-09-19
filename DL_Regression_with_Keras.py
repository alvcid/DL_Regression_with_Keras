import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.preprocessing import RobustScaler

boston_housing = datasets.boston_housing

print(boston_housing)

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

print(len(X_train))

# Visualize data
features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

df_train= pd.DataFrame(np.column_stack([X_train, y_train]), columns=features)
print(df_train)

# Divide data
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)
print(len(X_val))

# NN architecture
model = models.Sequential([
    tf.keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

print(model.summary())

# See the model deeply
print(model.layers)

hidden1 = model.layers[1]
weights, biases = hidden1.get_weights()

# compile the NN
model.compile(loss="mae",
              optimizer="Adam",
              metrics=["mae", "mse"])

# Data preparation for input shape
scaler = RobustScaler()

X_train_prep = scaler.fit_transform(X_train)
X_val_prep = scaler.transform(X_val)
X_test_prep = scaler.transform(X_test)

# Train the model
history = model.fit(X_train_prep, y_train,
          epochs=30,
          validation_data=(X_val_prep, y_val))

# plot the results
pd.DataFrame(history.history).plot(figsize=(10, 7))
plt.grid(True)
plt.xlabel("Epochs")
plt.ylabel("errors")
plt.show()

# validation
test_loss = model.evaluate(X_test_prep, y_test)
print("test_mse:", test_loss[1])

# Save the model
model.save("model_0")