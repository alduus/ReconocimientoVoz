# Escamilla Reséndiz Aldo - 2022630761
# Práctica Representación del habla e dominios del tiempo y frecuencia
# Reconocimiento de voz

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

audio_path = 'escuela.wav'
y, sr = librosa.load(audio_path, sr=None)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.title('Forma de onda (Sin divisiones)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.show()

segment_duration = 0.1
samples_per_segment = int(segment_duration * sr)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.title('Forma de onda (Dividida en segmentos de 100 ms)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')

total_duration = len(y) / sr
for t in np.arange(0, total_duration, segment_duration):
    plt.axvline(x=t, color='r', linestyle='--', alpha=0.7)

plt.show()

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(14, 6))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma de banda ancha')
plt.show()

D_narrow = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
plt.figure(figsize=(14, 6))
librosa.display.specshow(D_narrow, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma de banda estrecha')
plt.show()
