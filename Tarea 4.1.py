from scipy.io import wavfile as wav
import numpy as np
def generar_tonos(cromatica):
    s_rate=44100
    tiempo=np.arange(0,1,1/s_rate)
    cromatica=np.arange(1,13,1)
    melodia=[]
    for n in cromatica:
        for t in tiempo:
            frecuencia=440*(2**((n-10)/12))
            melodia.append(np.sin(2*np.pi*frecuencia*t))
    melodia=np.array(melodia)
    print(len(melodia))

    melodia_n= melodia*(2**15-1)/np.max(np.abs(melodia))
    wav.write('Tarea_1.wav', s_rate, melodia_n.astype(np.int16))
    def genera_fft(signal, s_rate):
        fft_out=abs(np.fft.fft(signal))
        freqs=np.fft.fftfreq(len(fft_out),1/s_rate)
        return fft_out[0:len(fft_out)//2], freqs[0:len(fft_out)//2]
    fft_out_p, freq_p=genera_fft(melodia,s_rate)
    s_rate, melodia = wav.read('Tarea_1.wav')

from scipy.io import wavfile as wav
import numpy as np
'''
def generar_tonos(cromatica, tiempos):
    s_rate = 44100  # frecuencia de muestreo (Hz)
    melodia = np.array([])

    for n, duracion in zip(cromatica, tiempos):
        # Genera un vector de tiempo de la duración específica
        tiempo = np.arange(0, duracion, 1/s_rate)
        
        # Calcula la frecuencia de la nota
        frecuencia = 440 * (2 ** ((n - 10) / 12))
        
        # Genera la onda seno para esa nota
        tono = np.sin(2 * np.pi * frecuencia * tiempo)
        
        # Concatenar el tono a la melodía total
        melodia = np.concatenate((melodia, tono))
    
    # Normalizar
    melodia_n = melodia * (2**15 - 1) / np.max(np.abs(melodia))
    melodia_n = melodia_n.astype(np.int16)
    
    # Guardar el archivo WAV
    wav.write('Piratas_del_Caribe.wav', s_rate, melodia_n)

    print("Archivo generado: Piratas_del_Caribe.wav")
    print(f"Duración total: {len(melodia) / s_rate:.2f} segundos")

# --- Ejemplo de uso ---

tiempos_caribe = [0.25,0.25,0.5,0.5,0.25,0.25,0.5,0.5,0.25,0.25,0.5,0.5,0.25,0.25,1]
'''
piratas_caribe = [5,8,10,10,10,12,13,13,13,15,12,10,8,8,10]
generar_tonos(piratas_caribe)
