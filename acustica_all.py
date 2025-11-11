#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 22:49:55 2020

@author: guanter

Modified on Thu Nov 06 2024 @ozezz
Included os path save
20241111: Removed simple audio, it is no longer maintined. Installation with conda arises problems
"""

#https://simpleaudio.readthedocs.io/en/latest/index.html

import numpy as np

import matplotlib, os
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
import scipy.signal
from scipy.signal import savgol_filter


plt.close('all')


#==============================================================================
#Modify the .rc file
# #==============================================================================

plt.rc('lines', linewidth=1., markersize=8)
plt.rc('grid', linewidth=0.5, ls='--', c='k', alpha=0.5)
plt.rc('xtick', direction='in',top='True',labelsize=10)
plt.rc('ytick', direction='in',right='True',labelsize=10)
plt.rc('font',family='serif')
plt.rc('legend', numpoints=1,)
plt.rc('axes.spines', top=True, right = True)



s_rate = 44100  # 44100 samples per second
#http://elclubdelautodidacta.es/wp/2012/08/calculo-de-la-frecuencia-de-nuestras-notas-musicales/

armonic_flg = False
stereo_flg = False
melodia_diat_flg = True
noise_flg = True

case_ind = 1


def genera_fft (signal, s_rate, str_nm, x_lim, x_lim_sub):
    
    fft_out = abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(fft_out), 1./s_rate)
    
    fft_out_p = fft_out[0:len(fft_out)//2]
    freq_p = freqs[0:len(fft_out)//2]
    
#    plt.figure(str_nm, figsize=(18, 5))
    fig, (ax) = plt.subplots(1,1)
    fig.set_size_inches(6.692913385826771, 4.136447244094488)

#    plt.subplot(1, 3, 2)
    ax.plot(freq_p, fft_out_p)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.xlim(x_lim)
    plt.grid()
    plt.savefig(os.path.join('Figures', '{filename}_fft.{extension}'.format(filename = str_nm, extension = 'png')),
                 bbox_inches='tight')
    plt.close()

    fig, (ax) = plt.subplots(1,1)
    fig.set_size_inches(6.692913385826771, 4.136447244094488)

    plt.plot(freq_p, fft_out_p)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.xlim(x_lim_sub)
    plt.grid()
    plt.savefig(os.path.join('Figures', '{filename}_fft_zoom.{extension}'.format(filename = str_nm, extension = 'png')),
                 bbox_inches='tight')
    plt.close()
    
    return fft_out_p, freq_p



def genera_nota (freq, t_nota, s_rate):

    t = np.linspace(0, t_nota, int(t_nota * s_rate), False)
    
    nota = np.sin(freq * t * 2 * np.pi)
    if armonic_flg == True:
        nota_1 = np.sin(0.5*freq * t * 2 * np.pi)
        nota_2 = np.sin(2* freq * t * 2 * np.pi)
        nota = 0.5 * nota + 0.25 * nota_1 + 0.25 * nota_2
    return nota


if case_ind == 1:
    
    t_nota = 1.  # Note duration of 3 seconds
    freq_a4 = 440 #LA 4
    ind_a4 = 9 #starting at 0
    n_notas = 12
    if melodia_diat_flg:
        ind_esc = [0, 2, 4, 5, 7, 9, 11] # melodia diatonica
        str_out = 'melodia_diatonica'
    else:        
        ind_esc = range(0, n_notas) # melodia cromática
        str_out = 'melodia_cromatica'

    rat = 2.**(1./12) #cond de octava: se dobla frecuencia en 12 notas

    ind_arr = np.arange(n_notas)
    freq_arr = freq_a4 * np.power(rat, ind_arr - ind_a4)

    freq_arr = freq_arr[ind_esc]
    print(freq_arr)
    time_arr = np.full(freq_arr.shape, t_nota)
    print(time_arr)
    

if case_ind == 2:    
    
#mirar pentagrama aquí https://es.wikipedia.org/wiki/Para_Elisa
    t_nota = 0.2  # Note duration of 3 seconds
    freq_arr = np.array([666, 627, 666, 627, 666, 494, 587, 523, 440, 262, 330, 440, 494])
    time_arr = np.full(freq_arr.shape, t_nota)
    time_arr[8] = 2*time_arr[8] 
    str_out = 'para_elisa'

if armonic_flg:
    str_out = str_out + '_arm'
                        
melodia = np.array([])
for freq, t_nota in zip(freq_arr, time_arr):
    nota = genera_nota (freq, t_nota, s_rate)
    melodia = np.append(melodia, nota)

print(melodia)
####Normalización de la melodia para entrar en el rango de int16
melodia = melodia * (2**15 - 1) / np.max(np.abs(melodia))  
print(melodia)  
melodia_int = melodia.astype(np.int16)

wav.write(os.path.join('wav_files', '{filename}.{extension}'.format(filename = str_out, extension = 'wav')),
           s_rate, melodia_int)

if case_ind == 1:
    fig, (ax) = plt.subplots(1,1)
    fig.set_size_inches(6.692913385826771, 4.136447244094488)
    ind_f = 0
    ax.plot(melodia[ind_f * s_rate: ind_f * s_rate + 1000], label='$f$=%.1f Hz'%(freq_arr[ind_f]))
    ind_f = len(freq_arr) -1
    ax.plot(melodia[ind_f * s_rate: ind_f * s_rate + 1000], label='$f$=%.1f Hz'%(freq_arr[ind_f]))

    ind_f = len(freq_arr) -3
    ax.plot(melodia[ind_f * s_rate: ind_f * s_rate + 1000], label='$f$=%.1f Hz'%(freq_arr[ind_f]))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend()
    plt.xlabel('Muestras (#)')
    plt.ylabel('Volumen')
    plt.grid()
    plt.savefig(os.path.join('Figures', '{filename}_vol.{extension}'.format(filename = str_out, extension = 'png')),
                 bbox_inches='tight')
    plt.close()


if stereo_flg == True:
    
    n_samp = len(melodia)
    audio = np.zeros((n_samp, 2))
    audio[:, 0] += melodia
    audio[:, 1] += 0.5 * melodia
        
    # normalize to 16-bit range
    audio *= (2**15 - 1) / np.max(np.abs(audio))
    # convert to 16-bit data
    audio = audio.astype(np.int16)
    
            
    wav.write(os.path.join('wav_files', '{filename}_stereo.{extension}'.format(filename = str_out, extension = 'wav')),
              s_rate, audio)


signal = np.copy(melodia)

x_lim = [freq_arr.min()-10, freq_arr.max()+10]
x_lim_sub = [freq_arr[5]-10, freq_arr[5]+10]
if armonic_flg:
    x_lim = [0.5*freq_arr.min()-10, 2*freq_arr.max()+10]
    x_lim_sub = [240, 280]
    
if case_ind == 2:
    x_lim = [150, 750]
    
fft_out_p, freq_p = genera_fft (signal, s_rate, 'fft_' + str_out, x_lim, x_lim_sub)


if noise_flg:
    
    def genera_ruido (signal, snr):
        noise = np.random.normal(0, 1, signal.shape)
        return signal + np.mean(np.abs(signal)) * noise / snr
        
    snr = 10
    
    signal_n = genera_ruido (signal, snr)
    fft_out_p, freq_p = genera_fft (signal_n, s_rate, 'fft_n_' + str_out, x_lim, x_lim_sub)
    
    signal_n_int = signal_n.astype(np.int16)
    print(os.path.join('wav_files', '{filename}_noise.{extension}'.format(filename = str_out, extension = 'wav')))
    wav.write(os.path.join('wav_files', '{filename}_noise.{extension}'.format(filename = str_out, extension = 'wav')),
                s_rate, signal_n_int)


    yhat = scipy.signal.savgol_filter(signal_n, 55, 3) # window size 51, polynomial order 3

    fig, (ax) = plt.subplots(1,1)
    fig.set_size_inches(6.692913385826771, 4.136447244094488)

    ind_f = 0
    plt.plot(signal_n[ind_f * s_rate: ind_f * s_rate + 1000], label='$f$=%.1f Hz'%(freq_arr[ind_f]))
    ind_f = len(freq_arr) -1
    plt.plot(signal_n[ind_f * s_rate: ind_f * s_rate + 1000], label='$f$=%.1f Hz'%(freq_arr[ind_f]))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid()
    plt.legend()
    plt.xlabel('Muestras (#)')
    plt.ylabel('Volumen')
    plt.savefig(os.path.join('Figures', '{filename}_vol_noise.{extension}'.format(filename = str_out, extension = 'png')), bbox_inches='tight')
    plt.close()

file_sound = os.path.join('wav_files', 'ParaElisa.wav')
rate, signal = wav.read(file_sound)

#signal = signal[30000:, :]
#plt.figure()
#plt.plot(signal[:, 0])
#signal = signal[signal>0]

t_1 = 0.68
t_2 = 3 + t_1
sub_audio = signal[int(t_1*rate):int(t_2*rate), :] #ya es int16


fft_out_p_wav, freq_p_wav = genera_fft (np.mean(sub_audio, axis=1), rate, 'fft_ParaElisa_mean', x_lim, x_lim_sub)
fft_out_p_wav1, freq_p_wav1 = genera_fft (sub_audio[:, 0], rate, 'fft_ParaElisa_canal_1', x_lim, x_lim_sub)
fft_out_p_wav2, freq_p_wav2 = genera_fft (sub_audio[:, 1], rate, 'fft_ParaElisa_canal_2', x_lim, x_lim_sub)

if case_ind == 2:

    fig, (ax) = plt.subplots(1,1)
    fig.set_size_inches(6.692913385826771, 4.136447244094488)
#    plt.subplot(1, 3, 2)
    plt.plot(freq_p, fft_out_p, label = 'manual')
    plt.plot(freq_p_wav, fft_out_p_wav*20., label = '.wav (x20)')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.xlim(x_lim)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join('Figures', '{filename}.{extension}'.format(filename = 'para_elisa_fft_comp', extension = 'png')),
                bbox_inches='tight')
    plt.close()

    fig, (ax) = plt.subplots(1,1)
    fig.set_size_inches(6.692913385826771, 4.136447244094488)
    plt.plot(freq_p_wav1, fft_out_p_wav1, label = 'canal 1')
    plt.plot(freq_p_wav2, fft_out_p_wav2, label = 'canal 2')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlim(x_lim)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join('Figures', '{filename}.{extension}'.format(filename = 'para_elisa_fft_comp_stereo', extension = 'png')),
                bbox_inches='tight')
    plt.close()

    tc = 0.3
    fig, (ax) = plt.subplots(1,1)
    fig.set_size_inches(6.692913385826771, 4.136447244094488)
    ind_m = int(tc * rate)
    plt.plot(np.linspace(t_1, tc + t_1, ind_m)+t_1, sub_audio[0:ind_m, 0], label='canal 1')
    plt.plot(np.linspace(t_1, tc + t_1, ind_m)+t_1, sub_audio[0:ind_m, 1], label='canal 2')
    plt.legend()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))    
    plt.grid()
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Volumen')
    plt.savefig(os.path.join('Figures', '{filename}.{extension}'.format(filename = 'para_elisa_wav_vol', extension = 'png')),
                 bbox_inches='tight')
    plt.close()


# Falta: combinar con armonicos y acordes
#
sub_audio = signal[int(t_1*rate):int(t_2*rate), :] #ya es int16


if noise_flg:
    
    def genera_ruido (signal, snr):
        noise = np.random.normal(0, 1, signal.shape)
        return signal + np.mean(np.abs(signal)) * noise / snr
        
    snr = 5
    
    sub_audio_t = np.ascontiguousarray(sub_audio[:, 1])
    sub_audio_n = genera_ruido (sub_audio_t, snr)

    sub_audio_sm = savgol_filter(sub_audio_n, 51, 3)
      
#    fft_out_p_wav_n, freq_p_wav_n = genera_fft (sub_audio_n, rate, 'fft_ParaElisa_mean_n', x_lim, x_lim_sub)
#    fft_out_p_wav_n = np.fft.fft(sub_audio_n)
#    fft_out_p_wav_n[fft_out_p_wav_n<0.1] = 0
#    fft_inv = np.fft.ifft(fft_out_p_wav_n)




    rate, signal = wav.read(file_sound)

    t_1 = 0.68
    t_2 = 3 + t_1
    sub_audio = signal[int(t_1*rate):int(t_2*rate), :] #ya es int16

    tc = 0.3


    fig, (ax) = plt.subplots(1,1)
    fig.set_size_inches(6.692913385826771, 4.136447244094488)
    
    ind_m = int(tc * rate)
    plt.plot(np.linspace(t_1, tc + t_1, ind_m)+t_1, sub_audio[0:ind_m, 0], label='canal 1')
    plt.plot(np.linspace(t_1, tc + t_1, ind_m)+t_1, sub_audio[0:ind_m, 1], label='canal 2')
    plt.legend()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))    
    plt.grid()
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Volumen')

    plt.savefig(os.path.join('Figures', '{filename}_vol_stereo.{extension}'.format(filename = str_out, extension = 'png')),
            bbox_inches='tight')
    plt.close()

    fig, (ax) = plt.subplots(1,1)
    fig.set_size_inches(6.692913385826771, 4.136447244094488)

    plt.plot(sub_audio_n[1000:1300], label='Con ruido')
    plt.plot(sub_audio_sm[1000:1300], label='Suavizado')
    plt.plot(sub_audio_t[1000:1300], '-r', label='Original')
    plt.xlabel('Muestras (#)')
    plt.ylabel('Volumen')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))    
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join('Figures', '{filename}.{extension}'.format(filename = 'para_elisa_suavizado', extension = 'png')),
             bbox_inches='tight')
    plt.close()
    

    wav.write(os.path.join('wav_files', '{filename}.{extension}'.format(filename = 'para_elisa_WAV', extension = 'wav')),
              s_rate, sub_audio.astype(np.int16))
    wav.write(os.path.join('wav_files', '{filename}.{extension}'.format(filename = 'para_elisa_WAV_noise', extension = 'wav')),
              s_rate, sub_audio_n.astype(np.int16))
    wav.write(os.path.join('wav_files', '{filename}.{extension}'.format(filename = 'para_elisa_WAV_sm', extension = 'wav')),
               s_rate, sub_audio_sm.astype(np.int16))
