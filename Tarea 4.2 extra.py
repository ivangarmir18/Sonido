#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Función completa `analizar(archivo_wav)` que realiza, directamente desde el .wav de entrada:
- lectura del archivo
- extracción de un recorte (t1=0.68 a t2=3+t1)
- cálculo y guardado de FFTs (mean, canal1, canal2) y versiones "zoom"
- graficado comparativo estéreo y volumen en los primeros instantes
- generación de versión con ruido, suavizado (savgol) y guardado de WAVs resultantes
- guarda las figuras en Sonido/Figures y los wav en Sonido/wav_files
Uso: analizar('Sonido/wav_files/piratas.wav')
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.signal import savgol_filter

# Ajustes estéticos (copiados del script original)
plt.rc('lines', linewidth=1., markersize=8)
plt.rc('grid', linewidth=0.5, ls='--', c='k', alpha=0.5)
plt.rc('xtick', direction='in', top=True, labelsize=10)
plt.rc('ytick', direction='in', right=True, labelsize=10)
plt.rc('font', family='serif')
plt.rc('legend', numpoints=1)
plt.rc('axes.spines', top=True, right=True)


def analizar(archivo_wav):
    """
    Analiza el wav dado y guarda las figuras/archivos resultantes.
    Parámetro:
        archivo_wav (str): ruta al fichero .wav a analizar
    """
    # --- Directorios de guardado (mantener compatibilidad con el script original) ---
    base_dir = 'Sonido'
    figures_dir = os.path.join(base_dir, 'Figures')
    wav_out_dir = os.path.join(base_dir, 'wav_files')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(wav_out_dir, exist_ok=True)

    # --- Funciones auxiliares (basadas en el original) ---
    def genera_fft(signal, s_rate, str_nm, x_lim, x_lim_sub):
        fft_out = abs(np.fft.fft(signal))
        freqs = np.fft.fftfreq(len(fft_out), 1. / s_rate)

        fft_out_p = fft_out[0:len(fft_out) // 2]
        freq_p = freqs[0:len(fft_out) // 2]

        # FFT completa (guardada)
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(6.692913385826771, 4.136447244094488)
        ax.plot(freq_p, fft_out_p)
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Amplitud')
        plt.xlim(x_lim)
        plt.grid()
        plt.savefig(os.path.join(figures_dir, f'{str_nm}_fft.png'), bbox_inches='tight')
        plt.close()

        # FFT zoom (guardada)
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(6.692913385826771, 4.136447244094488)
        ax.plot(freq_p, fft_out_p)
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Amplitud')
        plt.xlim(x_lim_sub)
        plt.grid()
        plt.savefig(os.path.join(figures_dir, f'{str_nm}_fft_zoom.png'), bbox_inches='tight')
        plt.close()

        return fft_out_p, freq_p

    def genera_ruido(signal_in, snr):
        noise = np.random.normal(0, 1, signal_in.shape)
        return signal_in + np.mean(np.abs(signal_in)) * noise / snr

    # --- Lectura del archivo ---
    if not os.path.isfile(archivo_wav):
        raise FileNotFoundError(f"No se encuentra el archivo: {archivo_wav}")
    rate, signal = wav.read(archivo_wav)
    nombre = os.path.splitext(os.path.basename(archivo_wav))[0]
    print(f"[analizar] Abriendo: {archivo_wav}  --  Fs = {rate} Hz  --  shape = {signal.shape}")

    # Asegurar formato (si es mono -> convertir a 2D consistente)
    if signal.ndim == 1:
        # mono a "estéreo" duplicando canal para facilitar el mismo procesamiento
        signal = np.stack([signal, signal], axis=1)
        print("[analizar] Archivo mono detectado: duplicando canal para análisis estéreo.")

    # Recorte temporal (igual que en el script original)
    t_1 = 0                       # principio
    t_2 = len(signal) / rate       # final en segundos
    sub_audio = signal[int(t_1*rate):int(t_2*rate), :]
    # int16 esperable

    # Parámetros de visualización de frecuencias (mismos para caso_ind==2 en el original)
    x_lim = [150, 750]
    x_lim_sub = [240, 280]

    # --- FFTs: media y canales ---
    fft_out_p_wav, freq_p_wav = genera_fft(np.mean(sub_audio, axis=1), rate, f'{nombre}_mean', x_lim, x_lim_sub)
    fft_out_p_wav1, freq_p_wav1 = genera_fft(sub_audio[:, 0].astype(float), rate, f'{nombre}_canal_1', x_lim, x_lim_sub)
    fft_out_p_wav2, freq_p_wav2 = genera_fft(sub_audio[:, 1].astype(float), rate, f'{nombre}_canal_2', x_lim, x_lim_sub)

    # --- Comparativa manual / "composición" como en el original ---
    # En el original comparaban la FFT manual (del tono generado) con la del .wav; aquí dejamos la comparativa de canales
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6.692913385826771, 4.136447244094488)
    plt.plot(freq_p_wav1, fft_out_p_wav1, label='canal 1')
    plt.plot(freq_p_wav2, fft_out_p_wav2, label='canal 2')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xlim(x_lim)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f'{nombre}_fft_comp_stereo.png'), bbox_inches='tight')
    plt.close()

    # --- Volumen (primeros instantes), tal y como en el script original ---
    tc = 0.3
    ind_m = int(tc * rate)
    if ind_m > sub_audio.shape[0]:
        ind_m = sub_audio.shape[0]
    t_axis = np.linspace(t_1, tc + t_1, ind_m)
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6.692913385826771, 4.136447244094488)
    plt.plot(t_axis, sub_audio[0:ind_m, 0], label='canal 1')
    plt.plot(t_axis, sub_audio[0:ind_m, 1], label='canal 2')
    plt.legend()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.grid()
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Volumen')
    plt.savefig(os.path.join(figures_dir, f'{nombre}_wav_vol.png'), bbox_inches='tight')
    plt.close()

    # --- Ruido y suavizado (savgol) sobre el canal derecho (como en el original) ---
    snr = 5
    sub_audio_t = np.ascontiguousarray(sub_audio[:, 1].astype(float))
    sub_audio_n = genera_ruido(sub_audio_t, snr)
    # Ajustar ventana para savgol: debe ser impar y menor que len
    win = 51
    if len(sub_audio_n) <= win:
        win = len(sub_audio_n) - 1 if (len(sub_audio_n) - 1) % 2 == 1 else len(sub_audio_n) - 2
        if win < 3:
            win = 3
    if win % 2 == 0:
        win += 1
    sub_audio_sm = savgol_filter(sub_audio_n, win, 3)

    # Graficar comparación ruido / suavizado / original
    slice_start = 1000
    slice_stop = 1300
    # asegurar índices dentro del rango
    N = len(sub_audio_t)
    slice_start = max(0, min(slice_start, N))
    slice_stop = max(0, min(slice_stop, N))
    if slice_start >= slice_stop:
        slice_start = 0
        slice_stop = min(300, N)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6.692913385826771, 4.136447244094488)
    plt.plot(sub_audio_n[slice_start:slice_stop], label='Con ruido')
    plt.plot(sub_audio_sm[slice_start:slice_stop], label='Suavizado')
    plt.plot(sub_audio_t[slice_start:slice_stop], '-r', label='Original')
    plt.xlabel('Muestras (#)')
    plt.ylabel('Volumen')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(figures_dir, f'{nombre}_suavizado.png'), bbox_inches='tight')
    plt.close()

    # --- Guardado de audios resultantes (normalizando a int16) ---
    def _normalize_to_int16(arr):
        # arr: float array
        if np.max(np.abs(arr)) == 0:
            return np.zeros_like(arr, dtype=np.int16)
        scaled = arr / np.max(np.abs(arr))
        return (scaled * (2**15 - 1)).astype(np.int16)

    # recorte original (sub_audio) -> si es stereo, lo guardamos; si no, creamos stereo duplicado (seguridad)
    try:
        wav.write(os.path.join(wav_out_dir, f'{nombre}_WAV.wav'), rate, sub_audio.astype(np.int16))
    except Exception:
        # si falla por tipos, normalizamos
        if sub_audio.dtype.kind == 'f':
            wav.write(os.path.join(wav_out_dir, f'{nombre}_WAV.wav'), rate, _normalize_to_int16(sub_audio))
        else:
            # for safety, ensure shape is (N,2)
            if sub_audio.ndim == 1:
                arr = np.stack([sub_audio, sub_audio], axis=1)
                wav.write(os.path.join(wav_out_dir, f'{nombre}_WAV.wav'), rate, arr.astype(np.int16))
            else:
                wav.write(os.path.join(wav_out_dir, f'{nombre}_WAV.wav'), rate, sub_audio.astype(np.int16))

    # ruido y suavizado (eran unidimensionales, canal derecho)
    wav.write(os.path.join(wav_out_dir, f'{nombre}_WAV_noise.wav'), rate, _normalize_to_int16(sub_audio_n))
    wav.write(os.path.join(wav_out_dir, f'{nombre}_WAV_sm.wav'), rate, _normalize_to_int16(sub_audio_sm))

    print(f"[analizar] Análisis completado. Figuras -> {figures_dir} ; WAVs -> {wav_out_dir}")

    # Ejemplo de uso:
    # analizar('Sonido/wav_files/ParaElisa.wav')
    # o
    # analizar('Sonido/wav_files/piratas.wav')

analizar('wav_files/piratas_del_caribe.wav')