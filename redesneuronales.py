import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.utils import class_weight


df = pd.read_csv("openpowerlifting-2022-12-31-a0e330e0.csv")

df['Squat1Kg'] = df['Squat1Kg'].mask(df['Squat1Kg'] < 0)
df['Bench1Kg'] = df['Bench1Kg'].mask(df['Bench1Kg'] < 0)
df['Deadlift1Kg'] = df['Deadlift1Kg'].mask(df['Deadlift1Kg'] < 0)
df['Squat2Kg'] = df['Squat2Kg'].mask(df['Squat2Kg'] < 0)
df['Bench2Kg'] = df['Bench2Kg'].mask(df['Bench2Kg'] < 0)
df['Deadlift2Kg'] = df['Deadlift2Kg'].mask(df['Deadlift2Kg'] < 0)
df['Squat3Kg'] = df['Squat3Kg'].mask(df['Squat3Kg'] < 0)
df['Bench3Kg'] = df['Bench3Kg'].mask(df['Bench3Kg'] < 0)
df['Deadlift3Kg'] = df['Deadlift3Kg'].mask(df['Deadlift3Kg'] < 0)

df.dropna(subset=['BodyweightKg'], inplace=True)

agrupado = df.groupby('WeightClassKg')
df['Squat1Kg'] = agrupado['Squat1Kg'].transform(lambda x: x.fillna(x.mean()))
df['Bench1Kg'] = agrupado['Bench1Kg'].transform(lambda x: x.fillna(x.mean()))
df['Deadlift1Kg'] = agrupado['Deadlift1Kg'].transform(lambda x: x.fillna(x.mean()))
df['Squat2Kg'] = agrupado['Squat2Kg'].transform(lambda x: x.fillna(x.mean()))
df['Bench2Kg'] = agrupado['Bench2Kg'].transform(lambda x: x.fillna(x.mean()))
df['Deadlift2Kg'] = agrupado['Deadlift2Kg'].transform(lambda x: x.fillna(x.mean()))
df['Squat3Kg'] = agrupado['Squat3Kg'].transform(lambda x: x.fillna(x.mean()))
df['Bench3Kg'] = agrupado['Bench3Kg'].transform(lambda x: x.fillna(x.mean()))
df['Deadlift3Kg'] = agrupado['Deadlift3Kg'].transform(lambda x: x.fillna(x.mean()))

df['Age'] = df['Age'].transform(lambda x: x.fillna(x.mean()))

df.dropna(subset=['Squat1Kg'], inplace=True)
df.dropna(subset=['Squat2Kg'], inplace=True)
df.dropna(subset=['Squat3Kg'], inplace=True)

df.dropna(subset=['Bench1Kg'], inplace=True)
df.dropna(subset=['Bench2Kg'], inplace=True)
df.dropna(subset=['Bench3Kg'], inplace=True)

df.dropna(subset=['Deadlift1Kg'], inplace=True)
df.dropna(subset=['Deadlift2Kg'], inplace=True)
df.dropna(subset=['Deadlift3Kg'], inplace=True)

df.drop(columns=['Deadlift4Kg', 'Squat4Kg','Bench4Kg'], inplace=True)

df['Best3BenchKg'] = df[['Bench1Kg', 'Bench2Kg', 'Bench3Kg']].max(axis=1)
df['Best3DeadliftKg'] = df[['Deadlift1Kg', 'Deadlift2Kg', 'Deadlift3Kg']].max(axis=1)
df['Best3SquatKg'] = df[['Squat1Kg', 'Squat2Kg', 'Squat3Kg']].max(axis=1)
df['TotalKg'] = df['Best3SquatKg'] + df['Best3BenchKg'] + df['Best3DeadliftKg']

df["Tested"] = df["Tested"].map(lambda x: "SI" if x == "Yes" else "NO")


# Seleccionar las variables independientes (características)
X = df[['Age','Sex_F', 'Sex_M', 'Sex_Mx','BodyweightKg','Tested','Equipment_Multi-ply','Equipment_Raw','Equipment_Single-ply','Equipment_Straps','Equipment_Unlimited','Equipment_Wraps']]  # Ajusta las columnas según las variables que desees incluir

# Seleccionar la variable objetivo
ySquat = df['Best3SquatKg']
yBench = df['Best3BenchKg']
yDeadlift = df['Best3DeadliftKg']

# Dividir en conjuntos de entrenamiento y prueba
X_trainSquat, X_testSquat, y_trainSquat, y_testSquat = train_test_split(X, ySquat, test_size=0.2, random_state=42)
X_trainBench, X_testBench, y_trainBench, y_testBench = train_test_split(X, yBench, test_size=0.2, random_state=42)
X_trainDeadlift, X_testDeadlift, y_trainDeadlift, y_testDeadlift = train_test_split(X, yDeadlift, test_size=0.2, random_state=42)


scaler = StandardScaler()

X_trainSquatScaled= scaler.fit_transform(X_trainSquat)
X_testSquatScaled= scaler.fit_transform(X_testSquat)
X_trainBenchScaled= scaler.fit_transform(X_trainBench)
X_testBenchScaled= scaler.fit_transform(X_testBench)
X_trainDeadliftScaled= scaler.fit_transform(X_trainDeadlift)
X_testDeadliftScaled= scaler.fit_transform(X_testDeadlift)



# Definir el modelo MLP
modelSquatMLP = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(X_trainSquatScaled.shape[1],), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Definir el optimizador con tasa de aprendizaje  0.00005
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)

# Compilar el modelo
modelSquatMLP.compile(optimizer=optimizer, loss='mean_squared_error')

# Entrenar el modelo  
modelSquatMLP.fit(X_trainSquatScaled, y_trainSquat, epochs=100, verbose=1)


# Realizar predicciones en el conjunto de prueba
predSquat = modelSquatMLP.predict(X_testSquatScaled)

# Supongamos que tienes los valores verdaderos en el tensor y_true y las predicciones en el tensor y_pred
predSquatFinal = tf.cast(tf.squeeze(predSquat), dtype=tf.float64)

# MAE (Mean Absolute Error)
mae = tf.reduce_mean(tf.abs(y_testSquat - predSquatFinal))
print("MAE:", mae.numpy())

# MSE (Mean Squared Error)
mse = tf.reduce_mean(tf.square(y_testSquat - predSquatFinal))
print("MSE:", mse.numpy())

# R2 (Coefficient of Determination)
total_sum_squares = tf.reduce_sum(tf.square(y_testSquat - tf.reduce_mean(y_testSquat)))
residual_sum_squares = tf.reduce_sum(tf.square(y_testSquat - predSquatFinal))
r2 = 1 - (residual_sum_squares / total_sum_squares)
print("R2:", r2.numpy())

# MAPE (Mean Absolute Percentage Error)
diff = tf.abs((y_testSquat - predSquatFinal) / tf.clip_by_value(tf.abs(y_testSquat), 1e-10, float('inf')))
mape = 100.0 * tf.reduce_mean(diff)
print("MAPE:", mape.numpy())



# Definir el modelo MLP
modelBenchMLP = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(X_trainBenchScaled.shape[1],), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Definir el optimizador con tasa de aprendizaje
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)

# Compilar el modelo
modelBenchMLP.compile(optimizer=optimizer, loss='mean_squared_error')

# Entrenar el modelo
modelBenchMLP.fit(X_trainBenchScaled, y_trainBench, epochs=100, verbose=1)

# Realizar predicciones en el conjunto de prueba
predBench = modelBenchMLP.predict(X_testBenchScaled)

# Supongamos que tienes los valores verdaderos en el tensor y_true y las predicciones en el tensor y_pred
predBenchFinal = tf.cast(tf.squeeze(predBench), dtype=tf.float64)

# MAE (Mean Absolute Error)
mae = tf.reduce_mean(tf.abs(y_testBench - predBenchFinal))
print("MAE:", mae.numpy())

# MSE (Mean Squared Error)
mse = tf.reduce_mean(tf.square(y_testBench - predBenchFinal))
print("MSE:", mse.numpy())

# R2 (Coefficient of Determination)
total_sum_squares = tf.reduce_sum(tf.square(y_testBench - tf.reduce_mean(y_testBench)))
residual_sum_squares = tf.reduce_sum(tf.square(y_testBench - predBenchFinal))
r2 = 1 - (residual_sum_squares / total_sum_squares)
print("R2:", r2.numpy())

# MAPE (Mean Absolute Percentage Error)
diff = tf.abs((y_testBench - predBenchFinal) / tf.clip_by_value(tf.abs(y_testBench), 1e-10, float('inf')))
mape = 100.0 * tf.reduce_mean(diff)
print("MAPE:", mape.numpy())


# Definir el modelo MLP
modelDeadliftMLP = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(X_trainDeadliftScaled.shape[1],), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Definir el optimizador con tasa de aprendizaje
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)

# Compilar el modelo
modelDeadliftMLP.compile(optimizer=optimizer, loss='mean_squared_error')

# Entrenar el modelo
modelDeadliftMLP.fit(X_trainDeadliftScaled, y_trainDeadlift, epochs=100, verbose=1)


# Realizar predicciones en el conjunto de prueba
predDeadlift = modelDeadliftMLP.predict(X_testDeadliftScaled)

# Supongamos que tienes los valores verdaderos en el tensor y_true y las predicciones en el tensor y_pred
predDeadliftFinal = tf.cast(tf.squeeze(predDeadlift), dtype=tf.float64)

# MAE (Mean Absolute Error)
mae = tf.reduce_mean(tf.abs(y_testDeadlift - predDeadliftFinal))
print("MAE:", mae.numpy())

# MSE (Mean Squared Error)
mse = tf.reduce_mean(tf.square(y_testDeadlift - predDeadliftFinal))
print("MSE:", mse.numpy())

# R2 (Coefficient of Determination)
total_sum_squares = tf.reduce_sum(tf.square(y_testDeadlift - tf.reduce_mean(y_testDeadlift)))
residual_sum_squares = tf.reduce_sum(tf.square(y_testDeadlift - predDeadliftFinal))
r2 = 1 - (residual_sum_squares / total_sum_squares)
print("R2:", r2.numpy())

# MAPE (Mean Absolute Percentage Error)
diff = tf.abs((y_testDeadlift - predDeadliftFinal) / tf.clip_by_value(tf.abs(y_testDeadlift), 1e-10, float('inf')))
mape = 100.0 * tf.reduce_mean(diff)
print("MAPE:", mape.numpy())

# Definir el modelo CNN
modelSquatCNN = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((X_trainSquatScaled.shape[1], 1), input_shape=(X_trainSquatScaled.shape[1],)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Definir el optimizador con tasa de aprendizaje
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

# Compilar el modelo
modelSquatCNN.compile(optimizer=optimizer, loss='mean_squared_error')

# Entrenar el modelo
modelSquatCNN.fit(X_trainSquatScaled, y_trainSquat, epochs=100, verbose=1)


# Realizar predicciones en el conjunto de prueba
predSquat = modelSquatCNN.predict(X_testSquatScaled)

# Supongamos que tienes los valores verdaderos en el tensor y_true y las predicciones en el tensor y_pred
predSquatFinal = tf.cast(tf.squeeze(predSquat), dtype=tf.float64)

# MAE (Mean Absolute Error)
mae = tf.reduce_mean(tf.abs(y_testSquat - predSquatFinal))
print("MAE:", mae.numpy())

# MSE (Mean Squared Error)
mse = tf.reduce_mean(tf.square(y_testSquat - predSquatFinal))
print("MSE:", mse.numpy())

# R2 (Coefficient of Determination)
total_sum_squares = tf.reduce_sum(tf.square(y_testSquat - tf.reduce_mean(y_testSquat)))
residual_sum_squares = tf.reduce_sum(tf.square(y_testSquat - predSquatFinal))
r2 = 1 - (residual_sum_squares / total_sum_squares)
print("R2:", r2.numpy())

# MAPE (Mean Absolute Percentage Error)
diff = tf.abs((y_testSquat - predSquatFinal) / tf.clip_by_value(tf.abs(y_testSquat), 1e-10, float('inf')))
mape = 100.0 * tf.reduce_mean(diff)
print("MAPE:", mape.numpy())


# Definir el modelo CNN
modelBenchCNN = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((X_trainBenchScaled.shape[1], 1), input_shape=(X_trainBenchScaled.shape[1],)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Definir el optimizador con tasa de aprendizaje
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

# Compilar el modelo
modelBenchCNN.compile(optimizer=optimizer, loss='mean_squared_error')

# Entrenar el modelo
modelBenchCNN.fit(X_trainBenchScaled, y_trainBench, epochs=100, verbose=1)


# Realizar predicciones en el conjunto de prueba
predBench = modelBenchCNN.predict(X_testBenchScaled)

# Supongamos que tienes los valores verdaderos en el tensor y_true y las predicciones en el tensor y_pred
predBenchFinal = tf.cast(tf.squeeze(predBench), dtype=tf.float64)

# MAE (Mean Absolute Error)
mae = tf.reduce_mean(tf.abs(y_testBench - predBenchFinal))
print("MAE:", mae.numpy())

# MSE (Mean Squared Error)
mse = tf.reduce_mean(tf.square(y_testBench - predBenchFinal))
print("MSE:", mse.numpy())

# R2 (Coefficient of Determination)
total_sum_squares = tf.reduce_sum(tf.square(y_testBench - tf.reduce_mean(y_testBench)))
residual_sum_squares = tf.reduce_sum(tf.square(y_testBench - predBenchFinal))
r2 = 1 - (residual_sum_squares / total_sum_squares)
print("R2:", r2.numpy())

# MAPE (Mean Absolute Percentage Error)
diff = tf.abs((y_testBench - predBenchFinal) / tf.clip_by_value(tf.abs(y_testBench), 1e-10, float('inf')))
mape = 100.0 * tf.reduce_mean(diff)
print("MAPE:", mape.numpy())



# Definir el modelo CNN
modelDeadliftCNN = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((X_trainDeadliftScaled.shape[1], 1), input_shape=(X_trainDeadliftScaled.shape[1],)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Definir el optimizador con tasa de aprendizaje
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

# Compilar el modelo
modelDeadliftCNN.compile(optimizer=optimizer, loss='mean_squared_error')

# Entrenar el modelo
modelDeadliftCNN.fit(X_trainDeadliftScaled, y_trainDeadlift, epochs=100, verbose=1)


# Realizar predicciones en el conjunto de prueba
predDeadlift = modelDeadliftCNN.predict(X_testDeadliftScaled)

# Supongamos que tienes los valores verdaderos en el tensor y_true y las predicciones en el tensor y_pred
predDeadliftFinal = tf.cast(tf.squeeze(predDeadlift), dtype=tf.float64)

# MAE (Mean Absolute Error)
mae = tf.reduce_mean(tf.abs(y_testDeadlift - predDeadliftFinal))
print("MAE:", mae.numpy())

# MSE (Mean Squared Error)
mse = tf.reduce_mean(tf.square(y_testDeadlift - predDeadliftFinal))
print("MSE:", mse.numpy())

# R2 (Coefficient of Determination)
total_sum_squares = tf.reduce_sum(tf.square(y_testDeadlift - tf.reduce_mean(y_testDeadlift)))
residual_sum_squares = tf.reduce_sum(tf.square(y_testDeadlift - predDeadliftFinal))
r2 = 1 - (residual_sum_squares / total_sum_squares)
print("R2:", r2.numpy())

# MAPE (Mean Absolute Percentage Error)
diff = tf.abs((y_testDeadlift - predDeadliftFinal) / tf.clip_by_value(tf.abs(y_testDeadlift), 1e-10, float('inf')))
mape = 100.0 * tf.reduce_mean(diff)
print("MAPE:", mape.numpy())


# Definir el modelo RNN
modelSquatRNN = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(64, input_shape=(X_trainSquatScaled.shape[1], 1), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Definir el optimizador con tasa de aprendizaje
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

# Compilar el modelo
modelSquatRNN.compile(optimizer=optimizer, loss='mean_squared_error')

# Entrenar el modelo
modelSquatRNN.fit(X_trainSquatScaled, y_trainSquat, epochs=100, verbose=1)


# Realizar predicciones en el conjunto de prueba
predSquat = modelSquatRNN.predict(X_testSquatScaled)

# Supongamos que tienes los valores verdaderos en el tensor y_true y las predicciones en el tensor y_pred
predSquatFinal = tf.cast(tf.squeeze(predSquat), dtype=tf.float64)

# MAE (Mean Absolute Error)
mae = tf.reduce_mean(tf.abs(y_testSquat - predSquatFinal))
print("MAE:", mae.numpy())

# MSE (Mean Squared Error)
mse = tf.reduce_mean(tf.square(y_testSquat - predSquatFinal))
print("MSE:", mse.numpy())

# R2 (Coefficient of Determination)
total_sum_squares = tf.reduce_sum(tf.square(y_testSquat - tf.reduce_mean(y_testSquat)))
residual_sum_squares = tf.reduce_sum(tf.square(y_testSquat - predSquatFinal))
r2 = 1 - (residual_sum_squares / total_sum_squares)
print("R2:", r2.numpy())

# MAPE (Mean Absolute Percentage Error)
diff = tf.abs((y_testSquat - predSquatFinal) / tf.clip_by_value(tf.abs(y_testSquat), 1e-10, float('inf')))
mape = 100.0 * tf.reduce_mean(diff)
print("MAPE:", mape.numpy())


# Definir el modelo RNN
modelBenchRNN = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(64, input_shape=(X_trainBenchScaled.shape[1], 1), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Definir el optimizador con tasa de aprendizaje
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

# Compilar el modelo
modelBenchRNN.compile(optimizer=optimizer, loss='mean_squared_error')

# Reshape de los datos de entrada para que coincida con el formato requerido por la RNN
X_train_rnn = X_trainBenchScaled.reshape(X_trainBenchScaled.shape[0], X_trainBenchScaled.shape[1], 1)

# Entrenar el modelo
modelBenchRNN.fit(X_train_rnn, y_trainBench, epochs=100, verbose=1)


# Realizar predicciones en el conjunto de prueba
predBench = modelBenchRNN.predict(X_testBenchScaled)

# Supongamos que tienes los valores verdaderos en el tensor y_true y las predicciones en el tensor y_pred
predBenchFinal = tf.cast(tf.squeeze(predBench), dtype=tf.float64)

# MAE (Mean Absolute Error)
mae = tf.reduce_mean(tf.abs(y_testBench - predBenchFinal))
print("MAE:", mae.numpy())

# MSE (Mean Squared Error)
mse = tf.reduce_mean(tf.square(y_testBench - predBenchFinal))
print("MSE:", mse.numpy())

# R2 (Coefficient of Determination)
total_sum_squares = tf.reduce_sum(tf.square(y_testBench - tf.reduce_mean(y_testBench)))
residual_sum_squares = tf.reduce_sum(tf.square(y_testBench - predBenchFinal))
r2 = 1 - (residual_sum_squares / total_sum_squares)
print("R2:", r2.numpy())

# MAPE (Mean Absolute Percentage Error)
diff = tf.abs((y_testBench - predBenchFinal) / tf.clip_by_value(tf.abs(y_testBench), 1e-10, float('inf')))
mape = 100.0 * tf.reduce_mean(diff)
print("MAPE:", mape.numpy())

# Definir el modelo RNN
modelDeadliftRNN = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(64, input_shape=(X_trainDeadliftScaled.shape[1], 1), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Definir el optimizador con tasa de aprendizaje
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

# Compilar el modelo
modelDeadliftRNN.compile(optimizer=optimizer, loss='mean_squared_error')

# Reshape de los datos de entrada para que coincida con el formato requerido por la RNN
X_train_rnn = X_trainDeadliftScaled.reshape(X_trainDeadliftScaled.shape[0], X_trainDeadliftScaled.shape[1], 1)

# Entrenar el modelo
modelDeadliftRNN.fit(X_train_rnn, y_trainDeadlift, epochs=100, verbose=1)

# Realizar predicciones en el conjunto de prueba
predDeadlift = modelDeadliftRNN.predict(X_testDeadliftScaled)

# Supongamos que tienes los valores verdaderos en el tensor y_true y las predicciones en el tensor y_pred
predDeadliftFinal = tf.cast(tf.squeeze(predDeadlift), dtype=tf.float64)

# MAE (Mean Absolute Error)
mae = tf.reduce_mean(tf.abs(y_testDeadlift - predDeadliftFinal))
print("MAE:", mae.numpy())

# MSE (Mean Squared Error)
mse = tf.reduce_mean(tf.square(y_testDeadlift - predDeadliftFinal))
print("MSE:", mse.numpy())

# R2 (Coefficient of Determination)
total_sum_squares = tf.reduce_sum(tf.square(y_testDeadlift - tf.reduce_mean(y_testDeadlift)))
residual_sum_squares = tf.reduce_sum(tf.square(y_testDeadlift - predDeadliftFinal))
r2 = 1 - (residual_sum_squares / total_sum_squares)
print("R2:", r2.numpy())

# MAPE (Mean Absolute Percentage Error)
diff = tf.abs((y_testDeadlift - predDeadliftFinal) / tf.clip_by_value(tf.abs(y_testDeadlift), 1e-10, float('inf')))
mape = 100.0 * tf.reduce_mean(diff)
print("MAPE:", mape.numpy())

