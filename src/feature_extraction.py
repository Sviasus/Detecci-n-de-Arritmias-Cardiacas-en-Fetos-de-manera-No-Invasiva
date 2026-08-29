import numpy as np
import pandas as pd
from scipy import signal


def calcular_tacograma_rr(picos_indices, fs=1000):
    """
    Convierte picos fQRS en tacograma RR y filtra artefactos de falso positivo/negativo.
    """
    if len(picos_indices) < 5:
        return np.array([])

    rr_crudo = (np.diff(picos_indices) / fs) * 1000.0

    # 1. Rango fisiológico fetal estricto (100 a 200 bpm -> 300 a 600 ms)
    rr_filtrado = []
    for r in rr_crudo:
        if 300.0 <= r <= 600.0:
            rr_filtrado.append(r)
        elif r > 600.0 and r <= 1200.0:
            # Latido perdido (falso negativo): dividir en dos intervalos iguales
            rr_filtrado.extend([r / 2.0, r / 2.0])
        # Picos espurios (< 300 ms) se descartan

    rr_arr = np.array(rr_filtrado)
    if len(rr_arr) < 5:
        return np.array([])

    # 2. Filtro de mediana móvil contra saltos abruptos (> 20% de diferencia local)
    rr_limpio = np.copy(rr_arr)
    w_size = 5
    for i in range(len(rr_arr)):
        i_s = max(0, i - w_size // 2)
        i_e = min(len(rr_arr), i + w_size // 2 + 1)
        med_local = np.median(rr_arr[i_s:i_e])

        if np.abs(rr_arr[i] - med_local) > (0.20 * med_local):
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
    Métricas en el dominio espectral mediante densidad espectral de Welch.
    """
    if len(rr_ms) < 10:
        return {"VLF": 0.0, "LF": 0.0, "HF": 0.0, "LF_HF_ratio": 0.0}

    tiempo_acum = np.cumsum(rr_ms) / 1000.0
    t_interp = np.arange(0, tiempo_acum[-1], 1.0 / fs_interp)
    rr_interp = np.interp(t_interp, tiempo_acum, rr_ms)
    rr_detrend = rr_interp - np.mean(rr_interp)

    nperseg = min(len(rr_detrend), int(fs_interp * 64))
    if nperseg < 8:
        nperseg = len(rr_detrend)

    freqs, psd = signal.welch(rr_detrend, fs=fs_interp, nperseg=nperseg)
    
    vlf_band = (freqs >= 0.0033) & (freqs < 0.04)
    lf_band = (freqs >= 0.04) & (freqs < 0.15)
    hf_band = (freqs >= 0.15) & (freqs < 0.40)

    vlf_pow = float(np.trapz(psd[vlf_band], freqs[vlf_band])) if np.any(vlf_band) else 0.0
    lf_pow = float(np.trapz(psd[lf_band], freqs[lf_band])) if np.any(lf_band) else 0.0
    hf_pow = float(np.trapz(psd[hf_band], freqs[hf_band])) if np.any(hf_band) else 0.0
    lf_hf = float(lf_pow / (hf_pow + 1e-8)) if hf_pow > 0 else 0.0

    return {
        "VLF": vlf_pow,
        "LF": lf_pow,
        "HF": hf_pow,
        "LF_HF_ratio": lf_hf
    }


def _calcular_sampen_numpy(serie, m=2, r=0.2):
    """
    Cálculo vectorial de Sample Entropy (SampEn) con NumPy.
    """
    N = len(serie)
    if N <= m + 1:
        return 0.0
    
    r_val = r * np.std(serie)
    if r_val == 0:
        return 0.0

    def _phi(dim):
        # Matriz de incrustación de dimensión dim
        x = np.array([serie[i:i + dim] for i in range(N - dim + 1)])
        # Distancia Chebyshev entre todos los pares de vectores
        d = np.max(np.abs(x[:, None, :] - x[None, :, :]), axis=2)
        # Contar pares con distancia <= r_val excluyendo auto-comparación
        count = np.sum(d <= r_val) - len(x)
        return count / (len(x) * (len(x) - 1))

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