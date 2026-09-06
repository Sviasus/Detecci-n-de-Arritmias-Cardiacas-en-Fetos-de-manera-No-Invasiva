import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.decomposition import FastICA

try:
    from preprocessing import preprocesar_senal_multicanal
except ImportError:
    from src.preprocessing import preprocesar_senal_multicanal


def cancelar_complejos_maternos(signals, fs=1000.0):
    """
    Cancela de forma adaptativa los complejos QRS maternos (mQRS) dominantes en las señales
    abdominales para evitar que contaminen la separación de fuentes ciegas de FastICA.
    Aplica una ventana suave de Hanning de +/- 45 ms alrededor de cada espiga materna.
    """
    signals = np.asarray(signals, dtype=np.float64)
    n_samples, n_channels = signals.shape
    signals_suprimidas = np.copy(signals)

    # 1. Identificar canal con mayor amplitud materna (envolvente RMS)
    canal_dominante_idx = int(np.argmax(np.std(signals, axis=0)))
    sig_ref = signals[:, canal_dominante_idx]

    # 2. Filtro pasa-banda para realzar la energía del QRS materno (8 a 25 Hz)
    nyq = 0.5 * fs
    b, a = butter(2, [8.0 / nyq, min(25.0 / nyq, 0.99)], btype="bandpass")
    sig_m_filt = filtfilt(b, a, sig_ref)

    # 3. Derivada y elevación al cuadrado (resalta las pendientes abruptas del QRS materno)
    sig_m_diff = np.gradient(sig_m_filt) ** 2
    w_integ = max(1, int(0.08 * fs))
    kernel = np.ones(w_integ) / w_integ
    sig_m_integ = np.convolve(sig_m_diff, kernel, mode="same")

    # 4. Detección de picos maternos (período refractario materno fisiológico ~450 ms -> max 130 bpm)
    dist_minima_m = int(0.42 * fs)
    umbral_m = np.percentile(sig_m_integ, 88)
    picos_m_integ, _ = find_peaks(sig_m_integ, height=umbral_m, distance=dist_minima_m)

    # 5. Refinamiento en la señal original para centrar exactamente el pico R materno
    w_search = int(0.040 * fs)
    picos_maternos = []
    for pm in picos_m_integ:
        ini = max(0, pm - w_search)
        fin = min(n_samples, pm + w_search)
        if fin > ini:
            p_extremo = ini + np.argmax(np.abs(sig_ref[ini:fin]))
            picos_maternos.append(p_extremo)

    picos_maternos = sorted(list(set(picos_maternos)))

    # 6. Atenuación adaptativa suave mediante ventana de Hanning invertida (+/- 45 ms)
    w_mask = int(0.045 * fs)
    len_mask = 2 * w_mask + 1
    # Máscara suave que atenúa el 85% de la energía del latido materno sin generar bordes abruptos
    mascara_suave = 1.0 - 0.85 * np.hanning(len_mask)

    for pm in picos_maternos:
        ini = max(0, pm - w_mask)
        fin = min(n_samples, pm + w_mask + 1)
        m_ini = ini - (pm - w_mask)
        m_fin = m_ini + (fin - ini)

        sub_mask = mascara_suave[m_ini:m_fin]
        for ch in range(n_channels):
            signals_suprimidas[ini:fin, ch] *= sub_mask

    return signals_suprimidas, picos_maternos


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

    # FastICA con pre-blanqueo y 3000 iteraciones para garantizar convergencia matemática estable
    ica = FastICA(
        n_components=n_components,
        random_state=42,
        max_iter=3000,
        tol=1e-3,
        whiten="unit-variance"
    )
    fuentes = ica.fit_transform(signals)

    # Puntuación combinada (Kurtosis + Densidad de picos en rango 110-220 bpm)
    mejores_scores = []
    nyq = 0.5 * fs
    b, a = butter(2, [10.0 / nyq, min(35.0 / nyq, 0.99)], btype="bandpass")

    for i in range(n_components):
        comp = fuentes[:, i]
        comp_filt = filtfilt(b, a, comp)
        
        # Kurtosis
        m = np.mean(comp_filt)
        s = np.std(comp_filt) + 1e-8
        kurt = np.mean(((comp_filt - m) / s) ** 4)
        
        # Densidad espectral en la banda fetal (1.8 - 3.5 Hz)
        freqs, psd = signal.welch(comp, fs=fs, nperseg=int(min(len(comp), 4 * fs)))
        idx_fetal = np.where((freqs >= 1.8) & (freqs <= 3.5))[0]
        potencia_fetal = np.sum(psd[idx_fetal]) / (np.sum(psd) + 1e-8)
        
        score = kurt * (1.0 + 2.5 * potencia_fetal)
        mejores_scores.append(score)

    mejor_idx = int(np.argmax(mejores_scores))
    fecg_aislado = fuentes[:, mejor_idx]

    # Corrección robusta de polaridad combinando skewness y percentiles extremos
    # Las ondas R fetales fisiológicas deben orientarse positivamente
    m = np.mean(fecg_aislado)
    s = np.std(fecg_aislado) + 1e-8
    skewness = np.mean(((fecg_aislado - m) / s) ** 3)
    q99 = np.percentile(fecg_aislado, 99)
    q01 = np.percentile(fecg_aislado, 1)

    if skewness < -0.2 or (np.abs(q01) > 1.25 * np.abs(q99)):
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


def extraer_fqrs_optimo(signals, fs=1000.0, return_intermediates=False):
    """
    Pipeline orquestador de extracción de complejos fQRS fetales a partir de registros multicanal.
    1. Preprocesamiento: Filtrado pasa-banda (1-45 Hz) y Notch (50/60 Hz) para supresión de ruido.
    2. Separación ciega de fuentes con FastICA y selección automática de componente fetal.
    3. Detección de cúspides de ondas R fetales mediante algoritmo Pan-Tompkins adaptativo.

    Parámetros:
        signals (np.ndarray): Matriz de señales multicanal (N_muestras, N_canales).
        fs (float): Frecuencia de muestreo en Hz (por defecto 1000 Hz).
        return_intermediates (bool): Si es True, retorna además (fecg_aislado, fuentes, idx_fetal, sig_filtrada).

    Retorna:
        np.ndarray: Índices muestrales de los picos fQRS detectados.
        (Opcional si return_intermediates=True): (picos, fecg_aislado, fuentes, idx_fetal, sig_filtrada)
    """
    signals = np.asarray(signals, dtype=np.float64)
    if len(signals) == 0:
        picos_vacio = np.array([], dtype=int)
        if return_intermediates:
            return picos_vacio, np.array([]), np.array([]), -1, np.array([])
        return picos_vacio

    # 1. Preprocesamiento multicanal (deriva de línea base y filtrado espectral)
    sig_filtrada = preprocesar_senal_multicanal(signals, fs=fs, aplicar_notch=True)

    # 2. Cancelación adaptativa de complejos maternos (mQRS) para liberar la fuente fetal
    sig_sin_materno, picos_maternos = cancelar_complejos_maternos(sig_filtrada, fs=fs)

    # 3. Aislamiento de la componente fetal por FastICA (sobre señal con mQRS suprimido)
    fecg_aislado, fuentes, idx_fetal = aislar_componente_fecg(sig_sin_materno, fs=fs)

    # 4. Detección Pan-Tompkins adaptativo sobre la componente fetal aislada
    picos_fqrs = detector_pan_tompkins_fetal(fecg_aislado, fs=fs)

    if return_intermediates:
        return picos_fqrs, fecg_aislado, fuentes, idx_fetal, sig_filtrada
    return picos_fqrs
