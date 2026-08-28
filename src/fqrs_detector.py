import numpy as np
import pandas as pd
from scipy import signal
from data_streamer import stream_cinc2013
from preprocessing import aplicar_filtros, extraer_componentes_ica


def suprimir_ecg_materno(sig, fs=1000):
    """
    Detecta los picos R maternos prominentes y atenúa la ventana del complejo mQRS.
    """
    sig_limpia = np.copy(sig)
    
    for ch in range(sig.shape[1]):
        canal = sig[:, ch]
        # Realce de picos maternos (muy anchos y altos)
        b, a = signal.butter(2, [8.0 / (0.5 * fs), 20.0 / (0.5 * fs)], btype='bandpass')
        canal_m = signal.filtfilt(b, a, canal)
        canal_m_sq = canal_m ** 2
        
        # Picos maternos: Frecuencia 50-100 bpm -> distancia min 500 ms
        picos_m, _ = signal.find_peaks(
            canal_m_sq, 
            height=np.percentile(canal_m_sq, 90), 
            distance=int(0.5 * fs)
        )
        
        # Blanking / atenuación suave del complejo materno (+/- 45 ms alrededor de cada pico)
        w_blank = int(0.045 * fs)
        for pm in picos_m:
            i_ini = max(0, pm - w_blank)
            i_fin = min(len(canal), pm + w_blank)
            # Suavizado por ventana Hanning para evitar saltos abruptos
            longitud = i_fin - i_ini
            if longitud > 0:
                ventana = signal.windows.hann(longitud)
                sig_limpia[i_ini:i_fin, ch] *= (1.0 - ventana)

    return sig_limpia


def pan_tompkins_fetal(canal, fs=1000):
    """
    Detector Pan-Tompkins adaptado a la morfología fetal rápida (110 - 160 bpm).
    """
    # 1. Filtro derivativo de 5 puntos
    h_der = np.array([-1, -2, 0, 2, 1]) * (fs / 8.0)
    derivada = np.convolve(canal, h_der, mode='same')

    # 2. Elevación al cuadrado
    cuadrado = derivada ** 2

    # 3. Integración por ventana móvil (~30 ms para fQRS)
    n_win = int(0.03 * fs)
    envolvente = np.convolve(cuadrado, np.ones(n_win) / n_win, mode='same')

    # 4. Umbral móvil adaptativo dual (ruido vs señal)
    distancia_min = int(0.26 * fs)  # Máximo ~230 bpm
    
    # Umbral por ventanas móviles de 2 segundos para tolerar cambios de amplitud
    win_samples = int(2.0 * fs)
    num_bloques = int(np.ceil(len(envolvente) / win_samples))
    todos_picos = []

    for b in range(num_bloques):
        ini = b * win_samples
        fin = min(len(envolvente), (b + 1) * win_samples)
        segmento = envolvente[ini:fin]
        
        if len(segmento) == 0:
            continue
            
        umbral_local = np.percentile(segmento, 70) * 0.5 + np.mean(segmento) * 0.5
        picos_seg, _ = signal.find_peaks(segmento, height=umbral_local, distance=distancia_min)
        todos_picos.extend(picos_seg + ini)

    picos_detectados = np.array(todos_picos, dtype=int)
    
    if len(picos_detectados) == 0:
        return np.array([])

    # 5. Ajuste de alineación con el pico original
    picos_alineados = []
    w_search = int(0.02 * fs)
    for p in picos_detectados:
        i_s = max(0, p - w_search)
        i_e = min(len(canal), p + w_search)
        idx_local = i_s + np.argmax(np.abs(canal[i_s:i_e]))
        picos_alineados.append(idx_local)

    return np.unique(np.array(picos_alineados, dtype=int))


def extraer_fqrs_optimo(sig_cruda, fs=1000):
    """
    Pipeline completo: Filtrado -> Supresión Materna -> FastICA -> Pan-Tompkins Multicanal.
    """
    sig_filt = aplicar_filtros(sig_cruda, fs)
    sig_sin_madre = suprimir_ecg_materno(sig_filt, fs)
    fuentes_ica = extraer_componentes_ica(sig_sin_madre)

    candidatos = []
    for ch in range(fuentes_ica.shape[1]):
        picos = pan_tompkins_fetal(fuentes_ica[:, ch], fs)
        if len(picos) < 15:
            continue

        # Evaluación fisiológica de la serie RR
        rr = np.diff(picos) / fs
        bpm = 60.0 / np.median(rr) if len(rr) > 0 else 0
        
        score = 0.0
        # Rango fisiológico normal fetal (110 - 165 bpm)
        if 110 <= bpm <= 165:
            score += 150.0
        elif 95 <= bpm <= 185:
            score += 80.0

        # Regularidad fisiológica
        iqr_rr = np.percentile(rr, 75) - np.percentile(rr, 25)
        if iqr_rr < 0.08:  # Menor a 80 ms de dispersión intercuartil
            score += 100.0

        candidatos.append({"picos": picos, "score": score, "bpm": bpm})

    if not candidatos:
        # Respaldo: aplicar directamente sobre el canal crudo de mayor varianza
        ch_max = np.argmax(np.var(sig_filt, axis=0))
        return pan_tompkins_fetal(sig_filt[:, ch_max], fs)

    candidatos.sort(key=lambda x: x["score"], reverse=True)
    return candidatos[0]["picos"]


def evaluar_fqrs(picos_detectados, picos_reales, fs=1000, tolerancia_ms=50):
    if picos_reales is None or len(picos_reales) == 0:
        return {"F1": 0.0, "TP": 0, "FP": 0, "FN": 0, "Sensibilidad": 0.0, "Precision": 0.0}

    tol_muestras = int((tolerancia_ms / 1000.0) * fs)
    picos_reales_restantes = list(picos_reales)
    
    tp = 0
    fp = 0

    for det in picos_detectados:
        match_idx = -1
        for i, ref in enumerate(picos_reales_restantes):
            if abs(det - ref) <= tol_muestras:
                match_idx = i
                break
        if match_idx != -1:
            tp += 1
            picos_reales_restantes.pop(match_idx)
        else:
            fp += 1

    fn = len(picos_reales_restantes)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * sensibilidad) / (precision + sensibilidad) if (precision + sensibilidad) > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision * 100,
        "Sensibilidad": sensibilidad * 100,
        "F1": f1 * 100
    }


def evaluar_dataset_completo_cinc2013(limite=75):
    resultados = []
    print(f"--- Evaluando CinC 2013 Set A con Pipeline Dual ({limite} registros) ---")

    for i in range(1, limite + 1):
        rec_name = f"a{i:02d}"
        try:
            sig_cruda, fs, fqrs_reales, _ = stream_cinc2013(rec_name)
            picos_detectados = extraer_fqrs_optimo(sig_cruda, fs)
            metricas = evaluar_fqrs(picos_detectados, fqrs_reales, fs)
            
            resultados.append({
                "Registro": rec_name,
                "Picos_Reales": len(fqrs_reales) if fqrs_reales is not None else 0,
                "Picos_Detectados": len(picos_detectados),
                "Sensibilidad": metricas["Sensibilidad"],
                "Precision": metricas["Precision"],
                "F1": metricas["F1"]
            })
            print(f"[{i:02d}/{limite}] {rec_name} -> F1: {metricas['F1']:.2f}% | Sens: {metricas['Sensibilidad']:.2f}% | Prec: {metricas['Precision']:.2f}%")
        except Exception as e:
            print(f"[{i:02d}/{limite}] {rec_name} -> Error: {e}")

    df_res = pd.DataFrame(resultados)
    
    print("\n=======================================================")
    print("           REPORTE GENERAL DE DESEMPEÑO fQRS          ")
    print("=======================================================")
    print(f"Registros evaluados con éxito : {len(df_res)}")
    print(f"Sensibilidad Promedio         : {df_res['Sensibilidad'].mean():.2f}%")
    print(f"Precisión Promedio            : {df_res['Precision'].mean():.2f}%")
    print(f"F1-Score Global Promedio      : {df_res['F1'].mean():.2f}%")
    print(f"Registros con F1 >= 90%       : {(df_res['F1'] >= 90.0).sum()} / {len(df_res)}")
    print("=======================================================")
    
    return df_res


if __name__ == "__main__":
    df_metricas = evaluar_dataset_completo_cinc2013(limite=75)