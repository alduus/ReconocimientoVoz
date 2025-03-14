import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Cargar los datos segmentados
df = pd.read_csv("segmentos_audio.csv")

# Asignar etiquetas manualmente (modificar según sea necesario)
etiquetas = ["S"] * 12 + ["V"] * 6 + ["S"] * 5 + ["U"] * 3 + ["S"] * 6
df = df.iloc[:len(etiquetas)].copy()
df["label"] = etiquetas

# Definir características y etiquetas
X = df[["mean_amplitude", "rms", "zero_crossing_rate"]]
y = df["label"]

# Normalizar los datos (estandarización)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Codificar etiquetas en valores numéricos (One-Hot Encoding)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = keras.utils.to_categorical(y_encoded)

# Dividir datos en entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Construir la red neuronal
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # 3 clases: S, U, V
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar la red neuronal
history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test))

# Evaluar el modelo en datos de prueba
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Precisión del modelo en datos de prueba: {test_acc:.2f}")

# Graficar la evolución del entrenamiento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.show()