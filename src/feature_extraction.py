import numpy as np
import pandas as pd
from scipy import signal


def calcular_tacograma_rr(picos_indices, fs=1000):
    """
    Convierte picos fQRS en tacograma RR respetando la fisiología fetal y preservando arritmias:
    - Rango fisiológico fetal ampliado: 230 ms a 860 ms (equivalente a 70 a 260 bpm).
    - Permite taquicardias supraventriculares (TPSV, 210-250 bpm) y bradicardias/pausas sinusales (70-90 bpm).
    - Elimina la división artificial de intervalos y evita la sobre-corrección de extrasístoles.
    """
    if len(picos_indices) < 5:
        return np.array([])

    rr_crudo = (np.diff(picos_indices) / fs) * 1000.0

    # 1. Rango fisiológico fetal admisible (70 a 260 bpm -> 230 a 860 ms)
    # Conserva la evidencia real de pausas, bloqueos AV y taquicardias fetales
    rr_fisiologico = [r for r in rr_crudo if 230.0 <= r <= 860.0]

    rr_arr = np.array(rr_fisiologico)
    if len(rr_arr) < 5:
        return np.array([])

    # 2. Filtrado selectivo de artefactos técnicos extremos (> 50% de variación local aislada)
    # Preserva las extrasístoles patológicas y las variaciones fisiológicas genuinas
    rr_limpio = np.copy(rr_arr)
    w_size = 7
    for i in range(len(rr_arr)):
        i_s = max(0, i - w_size // 2)
        i_e = min(len(rr_arr), i + w_size // 2 + 1)
        med_local = np.median(rr_arr[i_s:i_e])

        if np.abs(rr_arr[i] - med_local) > (0.50 * med_local):
            rr_limpio[i] = med_local

    return rr_limpio


def extraer_features_temporales(rr_ms):
    """
    Métricas en el dominio del tiempo: BPM, SDNN, RMSSD, pNN50.
    """
    if len(rr_ms) < 2:
        return {"BPM_mean": 0.0, "SDNN": 0.0, "RMSSD": 0.0, "pNN50": 0.0}
    
    bpm_inst = 60000.0 / rr_ms
    diff_rr = np.diff(rr_ms)
    
    return {
        "BPM_mean": float(np.mean(bpm_inst)),
        "SDNN": float(np.std(rr_ms, ddof=1)),
        "RMSSD": float(np.sqrt(np.mean(diff_rr ** 2))),
        "pNN50": float((np.sum(np.abs(diff_rr) > 50.0) / len(diff_rr)) * 100.0)
    }


def extraer_features_frecuenciales(rr_ms, fs_interp=4.0):
    """
    Métricas en el dominio espectral adaptadas a la fisiología fetal mediante densidad espectral de Welch.
    Bandas fetales canónicas:
    - VLF: < 0.04 Hz
    - LF: 0.04 - 0.20 Hz (modulación simpática y vasomotora fetal)
    - HF: 0.20 - 1.00 Hz (modulación vagal y movimientos respiratorios fetales)
    """
    if len(rr_ms) < 10:
        return {"VLF": 0.0, "LF": 0.0, "HF": 0.0, "LF_HF_ratio": 0.0}

    tiempo_acum = np.cumsum(rr_ms) / 1000.0
    t_interp = np.arange(0, tiempo_acum[-1], 1.0 / fs_interp)
    if len(t_interp) < 8:
        return {"VLF": 0.0, "LF": 0.0, "HF": 0.0, "LF_HF_ratio": 0.0}

    rr_interp = np.interp(t_interp, tiempo_acum, rr_ms)
    rr_detrend = rr_interp - np.mean(rr_interp)

    nperseg = min(len(rr_detrend), int(fs_interp * 64))
    if nperseg < 8:
        nperseg = len(rr_detrend)

    freqs, psd = signal.welch(rr_detrend, fs=fs_interp, nperseg=nperseg)
    
    # Bandas fetales validadas
    vlf_band = (freqs >= 0.0033) & (freqs < 0.04)
    lf_band = (freqs >= 0.04) & (freqs < 0.20)
    hf_band = (freqs >= 0.20) & (freqs < 1.00)

    # Integración trapezoidal robusta y compatible con scipy moderno
    def _integrar(y_vals, x_vals):
        if not np.any(y_vals):
            return 0.0
        try:
            from scipy.integrate import trapezoid
            return float(trapezoid(y_vals, x_vals))
        except ImportError:
            return float(np.trapz(y_vals, x_vals))

    vlf_pow = _integrar(psd[vlf_band], freqs[vlf_band])
    lf_pow = _integrar(psd[lf_band], freqs[lf_band])
    hf_pow = _integrar(psd[hf_band], freqs[hf_band])
    lf_hf = float(lf_pow / (hf_pow + 1e-8)) if hf_pow > 0 else 0.0

    return {
        "VLF": vlf_pow,
        "LF": lf_pow,
        "HF": hf_pow,
        "LF_HF_ratio": lf_hf
    }


def _calcular_sampen_numpy(serie, m=2, r=0.2):
    """
    Cálculo de Sample Entropy (SampEn) con NumPy optimizado en memoria por bloques.
    Reduce la memoria a O(chunk_size * K) para prevenir desbordamientos en series largas.
    """
    serie = np.asarray(serie, dtype=np.float64)
    N = len(serie)
    if N <= m + 1:
        return 0.0
    
    s_std = np.std(serie)
    if s_std < 1e-8:
        return 0.0
    r_val = r * s_std

    def _phi(dim):
        x = np.array([serie[i:i + dim] for i in range(N - dim + 1)])
        K = len(x)
        if K <= 1:
            return 0.0
        
        total_coincidencias = 0
        chunk_size = 256
        for start in range(0, K, chunk_size):
            end = min(K, start + chunk_size)
            chunk = x[start:end]
            diff = np.max(np.abs(chunk[:, None, :] - x[None, :, :]), axis=2)
            total_coincidencias += int(np.sum(diff <= r_val))
        
        total_coincidencias -= K
        return total_coincidencias / (K * (K - 1))

    try:
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        if phi_m > 0 and phi_m1 > 0:
            return float(-np.log(phi_m1 / phi_m))
        return 0.0
    except Exception:
        return 0.0



def _calcular_dfa_numpy(serie):
    """
    Cálculo de DFA (Detrended Fluctuation Analysis, exponente alpha) con NumPy.
    """
    N = len(serie)
    if N < 16:
        return 0.0
    
    # Serie integrada acumulativa centrada
    y = np.cumsum(serie - np.mean(serie))
    escalas = np.unique(np.logspace(np.log10(4), np.log10(max(5, N // 4)), num=8).astype(int))
    
    fluctuaciones = []
    escalas_validas = []

    for s in escalas:
        if s < 4 or s >= N:
            continue
        n_segmentos = N // s
        if n_segmentos < 2:
            continue

        f_s_total = 0.0
        for seg in range(n_segmentos):
            idx = slice(seg * s, (seg + 1) * s)
            t = np.arange(s)
            val = y[idx]
            # Ajuste lineal de tendencia
            p = np.polyfit(t, val, 1)
            tendencia = np.polyval(p, t)
            f_s_total += np.sum((val - tendencia) ** 2)

        f_s = np.sqrt(f_s_total / (n_segmentos * s))
        if f_s > 0:
            fluctuaciones.append(f_s)
            escalas_validas.append(s)

    if len(escalas_validas) < 3:
        return 0.0

    # Pendiente en escala log-log (exponente alpha)
    poly = np.polyfit(np.log(escalas_validas), np.log(fluctuaciones), 1)
    return float(poly[0])


def extraer_features_no_lineales(rr_ms):
    """
    Métricas no lineales: Poincaré (SD1, SD2), SampEn, DFA (alpha1).
    """
    if len(rr_ms) < 15:
        return {"SD1": 0.0, "SD2": 0.0, "SD1_SD2_ratio": 0.0, "SampEn": 0.0, "DFA_alpha1": 0.0}

    diff_rr = np.diff(rr_ms)
    var_rr = np.var(rr_ms, ddof=1)
    var_diff = np.var(diff_rr, ddof=1)
    
    sd1 = np.sqrt(0.5 * var_diff)
    sd2 = np.sqrt(max(0.0, 2 * var_rr - 0.5 * var_diff))
    sd1_sd2 = float(sd1 / sd2) if sd2 > 0 else 0.0

    sampen = _calcular_sampen_numpy(rr_ms)
    dfa_alpha = _calcular_dfa_numpy(rr_ms)

    return {
        "SD1": float(sd1),
        "SD2": float(sd2),
        "SD1_SD2_ratio": sd1_sd2,
        "SampEn": sampen,
        "DFA_alpha1": dfa_alpha
    }


def extraer_vector_caracteristicas_completo(picos_indices, fs=1000):
    """
    Calcula el vector consolidado de características fHRV.
    """
    rr_ms = calcular_tacograma_rr(picos_indices, fs)
    feats = {}
    feats.update(extraer_features_temporales(rr_ms))
    feats.update(extraer_features_frecuenciales(rr_ms))
    feats.update(extraer_features_no_lineales(rr_ms))
    feats["Num_Latidos_Validos"] = len(rr_ms)
    return feats