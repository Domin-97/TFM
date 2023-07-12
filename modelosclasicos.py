#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import tensorflow as tf


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

sample_weightsSquat = class_weight.compute_sample_weight(class_weight='balanced', y=y_trainSquat)

scaler = StandardScaler()

X_trainSquatScaled= scaler.fit_transform(X_trainSquat)
X_testSquatScaled= scaler.fit_transform(X_testSquat)
X_trainBenchScaled= scaler.fit_transform(X_trainBench)
X_testBenchScaled= scaler.fit_transform(X_testBench)
X_trainDeadliftScaled= scaler.fit_transform(X_trainDeadlift)
X_testDeadliftScaled= scaler.fit_transform(X_testDeadlift)


modelSquat = LinearRegression(fit_intercept= False)
modelBench = LinearRegression(fit_intercept= False)
modelDeadlift = LinearRegression(fit_intercept= False)

# Entrenar el modelo
modelSquat.fit(X_trainSquat, y_trainSquat,sample_weight=sample_weightsSquat)

# Realizar predicciones en el conjunto de prueba
predSquat = modelSquat.predict(X_testSquat)

# Calcular el MAE (Mean Absolute Error)
maeSquat = mean_absolute_error(y_testSquat, predSquat)
print("Mean Absolute Error (MAE):", maeSquat)

# Calcular el MSE (Mean Squared Error)
mseSquat = mean_squared_error(y_testSquat, predSquat)
print("Mean Squared Error (MSE):", mseSquat)

# Calcular el R2 (Coefficient of Determination)
r2Squat = r2_score(y_testSquat, predSquat)
print("Coefficient of Determination (R2):", r2Squat)

# Calcular el MAPE (Mean Absolute Percentage Error)
mapeSquat = np.mean(np.abs((y_testSquat - predSquat) / y_testSquat)) * 100
print("Mean Absolute Percentage Error (MAPE):", mapeSquat)



# Entrenar el modelo
modelBench.fit(X_trainBench, y_trainBench)

# Realizar predicciones en el conjunto de prueba
predBench = modelBench.predict(X_testBench)

# Calcular el MAE (Mean Absolute Error)
maeBench = mean_absolute_error(y_testBench, predBench)
print("Mean Absolute Error (MAE):", maeBench)

# Calcular el MSE (Mean Squared Error)
mseBench = mean_squared_error(y_testBench, predBench)
print("Mean Squared Error (MSE):", mseBench)

# Calcular el R2 (Coefficient of Determination)
r2Bench = r2_score(y_testBench, predBench)
print("Coefficient of Determination (R2):", r2Bench)

# Calcular el MAPE (Mean Absolute Percentage Error)
mapeBench = np.mean(np.abs((y_testBench - predBench) / y_testBench)) * 100
print("Mean Absolute Percentage Error (MAPE):", mapeBench)


# Entrenar el modelo
modelDeadlift.fit(X_trainDeadlift, y_trainDeadlift)

# Realizar predicciones en el conjunto de prueba
predDeadlift = modelDeadlift.predict(X_testDeadlift)

# Calcular el MAE (Mean Absolute Error)
maeDeadlift = mean_absolute_error(y_testDeadlift, predDeadlift)
print("Mean Absolute Error (MAE):", maeDeadlift)

# Calcular el MSE (Mean Squared Error)
mseDeadlift = mean_squared_error(y_testDeadlift, predDeadlift)
print("Mean Squared Error (MSE):", mseDeadlift)

# Calcular el R2 (Coefficient of Determination)
r2Deadlift = r2_score(y_testDeadlift, predDeadlift)
print("Coefficient of Determination (R2):", r2Deadlift)

# Calcular el MAPE (Mean Absolute Percentage Error)
mapeDeadlift = np.mean(np.abs((y_testDeadlift - predDeadlift) / y_testDeadlift)) * 100
print("Mean Absolute Percentage Error (MAPE):", mapeDeadlift)


modelSquat = KNeighborsRegressor(n_neighbors=7,weights='distance',metric='euclidean')
modelBench = KNeighborsRegressor(n_neighbors=7,weights='distance',metric='euclidean')
modelDeadlift = KNeighborsRegressor(n_neighbors=7,weights='distance',metric='euclidean')


# Entrenar el modelo
modelSquat.fit(X_trainSquatScaled, y_trainSquat)

# Realizar predicciones en el conjunto de prueba
predSquat = modelSquat.predict(X_testSquatScaled)

# Calcular el MAE (Mean Absolute Error)
maeSquat = mean_absolute_error(y_testSquat, predSquat)
print("Mean Absolute Error (MAE):", maeSquat)

# Calcular el MSE (Mean Squared Error)
mseSquat = mean_squared_error(y_testSquat, predSquat)
print("Mean Squared Error (MSE):", mseSquat)

# Calcular el R2 (Coefficient of Determination)
r2Squat = r2_score(y_testSquat, predSquat)
print("Coefficient of Determination (R2):", r2Squat)

# Calcular el MAPE (Mean Absolute Percentage Error)
mapeSquat = np.mean(np.abs((y_testSquat - predSquat) / y_testSquat)) * 100
print("Mean Absolute Percentage Error (MAPE):", mapeSquat)


# Entrenar el modelo
modelBench.fit(X_trainBenchScaled, y_trainBench)

# Realizar predicciones en el conjunto de prueba
predBench = modelBench.predict(X_testBenchScaled)

# Calcular el MAE (Mean Absolute Error)
maeBench = mean_absolute_error(y_testBench, predBench)
print("Mean Absolute Error (MAE):", maeBench)

# Calcular el MSE (Mean Squared Error)
mseBench = mean_squared_error(y_testBench, predBench)
print("Mean Squared Error (MSE):", mseBench)

# Calcular el R2 (Coefficient of Determination)
r2Bench = r2_score(y_testBench, predBench)
print("Coefficient of Determination (R2):", r2Bench)

# Calcular el MAPE (Mean Absolute Percentage Error)
mapeBench = np.mean(np.abs((y_testBench - predBench) / y_testBench)) * 100
print("Mean Absolute Percentage Error (MAPE):", mapeBench)


# Entrenar el modelo
modelDeadlift.fit(X_trainDeadliftScaled, y_trainDeadlift)

# Realizar predicciones en el conjunto de prueba
predDeadlift = modelDeadlift.predict(X_testDeadliftScaled)

# Calcular el MAE (Mean Absolute Error)
maeDeadlift = mean_absolute_error(y_testDeadlift, predDeadlift)
print("Mean Absolute Error (MAE):", maeDeadlift)

# Calcular el MSE (Mean Squared Error)
mseDeadlift = mean_squared_error(y_testDeadlift, predDeadlift)
print("Mean Squared Error (MSE):", mseDeadlift)

# Calcular el R2 (Coefficient of Determination)
r2Deadlift = r2_score(y_testDeadlift, predDeadlift)
print("Coefficient of Determination (R2):", r2Deadlift)

# Calcular el MAPE (Mean Absolute Percentage Error)
mapeDeadlift = np.mean(np.abs((y_testDeadlift - predDeadlift) / y_testDeadlift)) * 100
print("Mean Absolute Percentage Error (MAPE):", mapeDeadlift)


modelSquat = DecisionTreeRegressor(max_depth= 20,min_samples_split= 5,min_samples_leaf= 5,max_features= None)
modelBench = DecisionTreeRegressor(max_depth= 20,min_samples_split= 5,min_samples_leaf= 5,max_features= None)
modelDeadlift = DecisionTreeRegressor(max_depth= 20,min_samples_split= 5,min_samples_leaf= 5,max_features= None)

# Entrenar el modelo
modelSquat.fit(X_trainSquat, y_trainSquat)

# Realizar predicciones en el conjunto de prueba
predSquat = modelSquat.predict(X_testSquat)

# Calcular el MAE (Mean Absolute Error)
maeSquat = mean_absolute_error(y_testSquat, predSquat)
print("Mean Absolute Error (MAE):", maeSquat)

# Calcular el MSE (Mean Squared Error)
mseSquat = mean_squared_error(y_testSquat, predSquat)
print("Mean Squared Error (MSE):", mseSquat)

# Calcular el R2 (Coefficient of Determination)
r2Squat = r2_score(y_testSquat, predSquat)
print("Coefficient of Determination (R2):", r2Squat)

# Calcular el MAPE (Mean Absolute Percentage Error)
mapeSquat = np.mean(np.abs((y_testSquat - predSquat) / y_testSquat)) * 100
print("Mean Absolute Percentage Error (MAPE):", mapeSquat)

# Entrenar el modelo
modelBench.fit(X_trainBench, y_trainBench)

# Realizar predicciones en el conjunto de prueba
predBench = modelBench.predict(X_testBench)

# Calcular el MAE (Mean Absolute Error)
maeBench = mean_absolute_error(y_testBench, predBench)
print("Mean Absolute Error (MAE):", maeBench)

# Calcular el MSE (Mean Squared Error)
mseBench = mean_squared_error(y_testBench, predBench)
print("Mean Squared Error (MSE):", mseBench)

# Calcular el R2 (Coefficient of Determination)
r2Bench = r2_score(y_testBench, predBench)
print("Coefficient of Determination (R2):", r2Bench)

# Calcular el MAPE (Mean Absolute Percentage Error)
mapeBench = np.mean(np.abs((y_testBench - predBench) / y_testBench)) * 100
print("Mean Absolute Percentage Error (MAPE):", mapeBench)


# Entrenar el modelo
modelDeadlift.fit(X_trainDeadlift, y_trainDeadlift)

# Realizar predicciones en el conjunto de prueba
predDeadlift = modelDeadlift.predict(X_testDeadlift)

# Calcular el MAE (Mean Absolute Error)
maeDeadlift = mean_absolute_error(y_testDeadlift, predDeadlift)
print("Mean Absolute Error (MAE):", maeDeadlift)

# Calcular el MSE (Mean Squared Error)
mseDeadlift = mean_squared_error(y_testDeadlift, predDeadlift)
print("Mean Squared Error (MSE):", mseDeadlift)

# Calcular el R2 (Coefficient of Determination)
r2Deadlift = r2_score(y_testDeadlift, predDeadlift)
print("Coefficient of Determination (R2):", r2Deadlift)

# Calcular el MAPE (Mean Absolute Percentage Error)
mapeDeadlift = np.mean(np.abs((y_testDeadlift - predDeadlift) / y_testDeadlift)) * 100
print("Mean Absolute Percentage Error (MAPE):", mapeDeadlift)

