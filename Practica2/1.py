import wave
import numpy as np
import pandas as pd
import os

# Lista de archivos de audio (ajusta las rutas según tu sistema)
audio_files = {
    "parque": "parque.wav",
    "miperrosesaliodecasa": "miperrosesaliodecasa.wav",
    "lacasaesgrande": "lacasaesgrande.wav",
    "escuela": "escuela.wav",
    "casahogar": "casahogar.wav",
}

# Función para leer un archivo WAV y extraer segmentos
def process_audio(file_path, label, window_ms=100, sampling_rate=22050):
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

    # Definir tamaño de ventana en muestras
    window_size = int(window_ms * sampling_rate / 1000)

    # Extraer segmentos
    num_segments = len(audio_array) // window_size
    segments = []
    for i in range(num_segments):
        segment = audio_array[i * window_size: (i + 1) * window_size]
        segments.append({
            "audio_label": label,
            "segment_index": i,
            "mean_amplitude": np.mean(np.abs(segment)),  # Media de la amplitud absoluta
            "rms": np.sqrt(np.mean(segment**2)),  # Root Mean Square (RMS)
            "zero_crossing_rate": np.mean(np.diff(np.sign(segment)) != 0),  # Cruces por cero
        })

    return segments

# Procesar todos los archivos
segments_data = []
for label, file_path in audio_files.items():
    if os.path.exists(file_path):  # Verifica que el archivo exista
        segments_data.extend(process_audio(file_path, label))
    else:
        print(f"Advertencia: No se encontró el archivo {file_path}")

# Convertir datos a DataFrame y guardar como CSV
df_segments = pd.DataFrame(segments_data)
df_segments.to_csv("segmentos_audio.csv", index=False)

print("Proceso completado. Los datos han sido guardados en 'segmentos_audio.csv'.")