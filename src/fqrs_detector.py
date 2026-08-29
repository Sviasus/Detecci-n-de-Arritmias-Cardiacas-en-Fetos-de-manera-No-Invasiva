import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.decomposition import FastICA


def aislar_componente_fecg(signals, fs=1000.0, n_components=None):
    """
    Separa las fuentes bioeléctricas mediante FastICA y selecciona automáticamente
    la componente fetal basada en periodicidad y kurtosis.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]

    n_samples, n_channels = signals.shape
    if n_components is None:
        n_components = min(n_channels, 4)

    # FastICA con pre-blanqueo
    ica = FastICA(
        n_components=n_components,
        random_state=42,
        max_iter=1000,
        tol=1e-3,
        whiten="unit-variance"
    )
    fuentes = ica.fit_transform(signals)

    # Puntuación combinada (Kurtosis + Densidad de picos en rango 110-200 bpm)
    mejores_scores = []
    nyq = 0.5 * fs
    b, a = butter(2, [10.0 / nyq, 35.0 / nyq], btype="bandpass")

    for i in range(n_components):
        comp = fuentes[:, i]
        comp_filt = filtfilt(b, a, comp)
        
        # Kurtosis
        m = np.mean(comp_filt)
        s = np.std(comp_filt) + 1e-8
        kurt = np.mean(((comp_filt - m) / s) ** 4)
        
        # Densidad espectral en la banda fetal (1.8 - 3.3 Hz)
        freqs, psd = signal.welch(comp, fs=fs, nperseg=int(4 * fs))
        idx_fetal = np.where((freqs >= 1.8) & (freqs <= 3.5))[0]
        potencia_fetal = np.sum(psd[idx_fetal]) / (np.sum(psd) + 1e-8)
        
        score = kurt * (1.0 + 2.0 * potencia_fetal)
        mejores_scores.append(score)

    mejor_idx = int(np.argmax(mejores_scores))
    fecg_aislado = fuentes[:, mejor_idx]

    # Corrección estricta de polaridad: las espigas QRS deben ser positivas
    q99 = np.percentile(fecg_aislado, 99)
    q01 = np.percentile(fecg_aislado, 1)
    if np.abs(q01) > np.abs(q99):
        fecg_aislado = -fecg_aislado

    return fecg_aislado, fuentes, mejor_idx


def detector_pan_tompkins_fetal(fecg_signal, fs=1000.0):
    """
    Algoritmo Pan-Tompkins optimizado con umbral adaptativo por ventanas temporales
    para detectar la totalidad de complejos fQRS sin pérdidas por atenuación.
    """
    fecg_signal = np.asarray(fecg_signal, dtype=np.float64).flatten()
    n_total = len(fecg_signal)

    # 1. Filtro Pasa-Banda específico fQRS (10 - 35 Hz)
    nyq = 0.5 * fs
    low = 10.0 / nyq
    high = min(35.0 / nyq, 0.99)
    b, a = butter(2, [low, high], btype="bandpass")
    sig_band = filtfilt(b, a, fecg_signal)

    # 2. Derivada de cinco puntos (acentúa pendientes rápidas de despolarización)
    sig_diff = np.gradient(sig_band)

    # 3. Elevación al cuadrado
    sig_sq = sig_diff ** 2

    # 4. Integración por media móvil (~70 ms para feto)
    w_len = max(1, int(0.07 * fs))
    kernel = np.ones(w_len) / w_len
    sig_integ = np.convolve(sig_sq, kernel, mode="same")

    # 5. Detección por ventanas adaptativas de 3 segundos
    # Período refractario fisiológico fetal: ~240 ms (frecuencia cardíaca máx ~250 bpm)
    dist_minima = int(0.24 * fs)
    win_size = int(3.0 * fs)
    todos_picos_integ = []

    for start in range(0, n_total, win_size):
        end = min(start + win_size, n_total)
        segmento = sig_integ[start:end]
        
        if len(segmento) < dist_minima:
            continue
            
        # Umbral dinámico adaptativo local (media + fracción del pico del segmento)
        mediana_local = np.median(segmento)
        max_local = np.percentile(segmento, 95)
        umbral_local = mediana_local + 0.25 * (max_local - mediana_local)
        
        picos_seg, _ = find_peaks(segmento, height=umbral_local, distance=dist_minima)
        todos_picos_integ.extend(start + picos_seg)

    todos_picos_integ = sorted(list(set(todos_picos_integ)))

    # 6. Refinamiento en señal original para fijar la cúspide exacta de la onda R (+/- 35 ms)
    w_search = int(0.035 * fs)
    picos_fQRS_finales = []

    for p in todos_picos_integ:
        ini = max(0, p - w_search)
        fin = min(n_total, p + w_search)
        if fin > ini:
            p_max = ini + np.argmax(fecg_signal[ini:fin])
            picos_fQRS_finales.append(p_max)

    # Eliminar duplicados cercanos que violen el período refractario
    picos_filtrados = []
    for p in sorted(list(set(picos_fQRS_finales))):
        if not picos_filtrados or (p - picos_filtrados[-1]) >= dist_minima:
            picos_filtrados.append(p)

    return np.array(picos_filtrados, dtype=int)