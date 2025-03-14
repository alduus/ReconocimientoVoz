import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Cargar datos procesados (debes haber generado 'segmentos_audio.csv' antes)
df = pd.read_csv("segmentos_audio.csv")

# Agregar etiquetas manualmente (modifica según sea necesario)
etiquetas = ["S"] * 12 + ["V"] * 6 + ["S"] * 5 + ["U"] * 3 + ["S"] * 6  # Ajusta según la gráfica

# Asegurar que el número de etiquetas coincide con los segmentos
df = df.iloc[:len(etiquetas)].copy()  # Tomar solo las filas necesarias
df["label"] = etiquetas  # Asignar etiquetas

# Definir características y etiquetas
X = df[["mean_amplitude", "rms", "zero_crossing_rate"]]
y = df["label"]

# Dividir datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el árbol de decisión
clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluar precisión en datos de prueba
accuracy = clf.score(X_test, y_test)
print(f"Precisión del modelo: {accuracy:.2f}")

# Graficar el árbol de decisión
plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=X.columns, class_names=["S", "U", "V"], filled=True)
plt.title("Árbol de Decisión para Clasificación de Segmentos de Audio")
plt.show()