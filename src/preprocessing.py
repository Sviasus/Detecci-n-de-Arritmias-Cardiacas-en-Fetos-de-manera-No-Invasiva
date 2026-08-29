import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from sklearn.decomposition import FastICA


def filtro_butter_pasa_banda(data, lowcut=1.0, highcut=45.0, fs=1000.0, order=3):
    nyquist = 0.5 * fs
    low = max(1e-4, min(lowcut / nyquist, 0.99))
    high = max(low + 1e-4, min(highcut / nyquist, 0.99))
    b, a = butter(order, [low, high], btype='bandpass')
    return filtfilt(b, a, data, axis=0)


def filtro_notch(data, f0=50.0, Q=30.0, fs=1000.0):
    nyquist = 0.5 * fs
    w0 = f0 / nyquist
    if 0 < w0 < 1:
        b, a = iirnotch(w0, Q)
        return filtfilt(b, a, data, axis=0)
    return data


def remover_deriva_linea_base(senal, fs=1000.0):
    w1 = int(0.2 * fs)
    if w1 % 2 == 0:
        w1 += 1
    w2 = int(0.6 * fs)
    if w2 % 2 == 0:
        w2 += 1
        
    if len(senal) > w2:
        base1 = medfilt(senal, kernel_size=w1)
        linea_base = medfilt(base1, kernel_size=w2)
        return senal - linea_base
    return senal - np.mean(senal)


def normalizar_zscore(senal):
    desv = np.nanstd(senal)
    if desv > 1e-8:
        return (senal - np.nanmean(senal)) / desv
    return senal - np.nanmean(senal)


def preprocesar_senal_multicanal(signals, fs=1000.0, aplicar_notch=True, normalizar=False):
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]
        
    n_samples, n_channels = signals.shape
    signals_proc = np.zeros_like(signals)
    
    for ch in range(n_channels):
        raw_ch = signals[:, ch]
        ch_centered = raw_ch - np.nanmean(raw_ch)
        ch_no_drift = remover_deriva_linea_base(ch_centered, fs=fs)
        ch_band = filtro_butter_pasa_banda(ch_no_drift, lowcut=1.0, highcut=45.0, fs=fs, order=3)
        
        if aplicar_notch:
            ch_n50 = filtro_notch(ch_band, f0=50.0, Q=30.0, fs=fs)
            ch_clean = filtro_notch(ch_n50, f0=60.0, Q=30.0, fs=fs)
        else:
            ch_clean = ch_band
            
        if normalizar:
            signals_proc[:, ch] = normalizar_zscore(ch_clean)
        else:
            signals_proc[:, ch] = ch_clean
            
    return signals_proc


# Alias y funciones de compatibilidad
def aplicar_filtros(signals, fs=1000.0):
    return preprocesar_senal_multicanal(signals, fs=fs)


def extraer_componentes_ica(signals, n_components=None, random_state=42):
    if n_components is None:
        n_components = min(signals.shape[1], 4)
    ica = FastICA(n_components=n_components, random_state=random_state, max_iter=1000, tol=1e-3)
    return ica.fit_transform(signals)