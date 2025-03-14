import numpy as np
import matplotlib.pyplot as plt
import wave
import pandas as pd

# Función para cargar el audio y segmentarlo
def load_audio(file_path, sampling_rate=22050, window_ms=100):
    with wave.open(file_path, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        num_frames = wf.getnframes()
        audio_data = wf.readframes(num_frames)

    # Convertir datos binarios a array numpy
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Ajustar tasa de muestreo si es necesario
    if frame_rate != sampling_rate:
        num_samples = int(len(audio_array) * sampling_rate / frame_rate)
        audio_array = np.interp(
            np.linspace(0, len(audio_array), num_samples),
            np.arange(len(audio_array)),
            audio_array
        )

    return audio_array, sampling_rate

# Cargar el archivo de audio
file_path = "escuela.wav"  # Ajusta el nombre del archivo
audio, sr = load_audio(file_path)

# Definir los segmentos (100 ms)
window_size = int(0.1 * sr)
num_segments = len(audio) // window_size
times = np.linspace(0, len(audio) / sr, len(audio))

# Crear etiquetas manualmente para cada segmento
labels = ["S"] * 12 + ["V"] * 6 + ["S"] * 5 + ["U"] * 3 + ["S"] * 6  # Ajusta según necesites

# Graficar la forma de onda segmentada
plt.figure(figsize=(12, 4))
plt.plot(times, audio, label="Forma de onda", color="blue")

# Dibujar líneas divisorias y etiquetas
for i in range(num_segments):
    t = i * 0.1  # Convertir índice de segmento a segundos
    plt.axvline(x=t, color="red", linestyle="dashed", linewidth=0.5)
    if i < len(labels):  # Asegurar que hay etiquetas suficientes
        plt.text(t + 0.02, -0.6, labels[i], fontsize=12, fontweight="bold")

# Configurar ejes y título
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title(f"Forma de onda - {file_path} (Dividida en segmentos de 100 ms)")
plt.show()