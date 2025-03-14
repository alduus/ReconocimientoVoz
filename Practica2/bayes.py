import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Cargar los datos procesados
df = pd.read_csv("segmentos_audio.csv")

# Asignar etiquetas manualmente (ajustar según sea necesario)
etiquetas = ["S"] * 12 + ["V"] * 6 + ["S"] * 5 + ["U"] * 3 + ["S"] * 6  
df = df.iloc[:len(etiquetas)].copy()
df["label"] = etiquetas  # Agregar etiquetas

# Definir las características y la variable objetivo
X = df[["mean_amplitude", "rms", "zero_crossing_rate"]]
y = df["label"]

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo Naïve Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Hacer predicciones
y_pred = nb_classifier.predict(X_test)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

# Reporte de clasificación
print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))

# Matriz de confusión
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(nb_classifier, X_test, y_test, cmap="Blues")
plt.title("Matriz de Confusión - Naïve Bayes")
plt.show()