#%%
# Reading wav file
import soundfile
import numpy as np
import pandas as pd
from scipy.fft import fft
from PyEMD import EMD
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import librosa, librosa.display


# based on https://towardsdatascience.com/improve-your-time-series-analysis-with-stochastic-and-deterministic-components-decomposition-464e623f8270
def emd(signal, plot=False):
    emd = EMD(DTYPE=np.float16, spline_kind='akima')
    imfs = emd(signal.values)

    if plot:
        t = [i for i in range(len(signal))]
        N = imfs.shape[0]
        fig, axs = plt.subplots(N + 1, 1, figsize=(15, 9))
        axs[0].plot(t, signal)
        axs[0].set_title('Original Signal')
        for n, imf in enumerate(imfs):
            axs[n + 1].plot(t, imf)
            axs[n + 1].set_title(f'IMF {n}')
        plt.show()

    return imfs


def phase_spectrum(imfs, plot=False):
    imfs_p = []
    for i, imf in enumerate(imfs):
        trans = fft(imf)
        imf_p = np.arctan(trans.imag / trans.real)
        imfs_p.append(imf_p)

    if plot:
        fig, axs = plt.subplots(len(imfs), 1, figsize=(15, 9))
        for i, imf in enumerate(imf_p):
            axs[i].plot(imf_p, 'o')
            axs[i].set_title(f'IMF {i}')
        plt.show()

    mis = []
    for i in range(len(imfs_p) - 1):
        mis.append(mutual_info_regression(imfs_p[i].reshape(-1, 1), imfs_p[i + 1])[0])
    mis = np.array(mis)

    return imfs_p, mis


def divide_signal(signal, imfs, mis, cutoff=0.05, plot=True):
    cut_point = np.where(mis > cutoff)[0][0]
    stochastic_component = np.sum(imfs[:cut_point], axis=0)
    deterministic_component = np.sum(imfs[cut_point:], axis=0)

    t = [i for i in range(len(signal))]

    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(15, 12))
        axs[0].plot(t, signal.values)
        axs[0].set_title('Original Signal')

        axs[1].plot(t, stochastic_component)
        axs[1].set_title('Stochastic Component')

        axs[2].plot(t, deterministic_component)
        axs[2].set_title('Deterministic Component')
        plt.show()

    return stochastic_component, deterministic_component


def plot_signal(x, sr=8000, name='signal'):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(x, sr=sr)
    plt.title(f'{name}')
    plt.show()


def spec_plot(s, ms=93, sr=8000, name='signal'):
    n_fft = int(2**round(np.log2((ms/1000)*sr)))
    h = int(n_fft/2)
    S = librosa.stft(s, n_fft=n_fft, hop_length=h)
    # S = np.abs(S)
    # S = librosa.power_to_db(S)
    S = librosa.amplitude_to_db(S)
    S = np.abs(S)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='hz')
    plt.title(f'{name}')
    plt.colorbar()
    plt.show()


def save_audio(x, sr=8000, name='signal'):
    soundfile.write(f'{name}.wav', x, sr, subtype='PCM_24')


def get_components(s, cutoff=0.005):
    s_pandas = pd.Series(s)
    imfs = emd(s_pandas, plot=True)
    phases, mis = phase_spectrum(imfs)
    sc, dc = divide_signal(s_pandas, imfs, mis, cutoff=cutoff)
    return sc, dc, imfs


## add noise
def add_white_noise(x, alpha=0.05):
    noise = np.random.normal(0, np.std(x), x.shape) * alpha
    return x+noise

## standardization
def standardization(x):
    return (x-np.mean(x))/np.std(x)