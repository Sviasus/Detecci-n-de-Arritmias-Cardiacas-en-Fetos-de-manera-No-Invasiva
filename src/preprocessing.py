import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from data_streamer import stream_cinc2013


def limpiar_nans(sig):
    """
    Rellena valores NaN o Inf usando interpolación lineal por columna.
    """
    df = pd.DataFrame(sig)
    # Interpolar hacia adelante y hacia atrás para bordes
    df = df.interpolate(method='linear', limit_direction='both', axis=0)
    df = df.fillna(0.0)
    return df.to_numpy()


def aplicar_filtros(sig, fs=1000):
    """
    Limpia NaNs y aplica filtro pasa-banda (1-100 Hz) y Notch (50 y 60 Hz).
    """
    # 0. Limpieza previa de NaNs
    sig_limpia = limpiar_nans(sig)

    # 1. Pasa-banda Butterworth orden 3 (1 a 100 Hz)
    nyq = 0.5 * fs
    low = 1.0 / nyq
    high = 100.0 / nyq
    b_band, a_band = signal.butter(3, [low, high], btype='bandpass')
    sig_band = signal.filtfilt(b_band, a_band, sig_limpia, axis=0)

    # 2. Notch 50 Hz y 60 Hz
    for f0 in [50.0, 60.0]:
        w0 = f0 / nyq
        Q = 30.0
        b_notch, a_notch = signal.iirnotch(w0, Q)
        sig_band = signal.filtfilt(b_notch, a_notch, sig_band, axis=0)

    return sig_band


def extraer_componentes_ica(sig_filtrada, random_state=42):
    """
    Aplica FastICA a los 4 canales garantizando datos finitos.
    """
    # Verificación de seguridad de finitud
    sig_filtrada = np.nan_to_num(sig_filtrada, nan=0.0, posinf=0.0, neginf=0.0)
    ica = FastICA(n_components=4, random_state=random_state, max_iter=1000, tol=1e-4)
    fuentes_ica = ica.fit_transform(sig_filtrada)
    return fuentes_ica


def seleccionar_canal_fetal(fuentes_ica, fs=1000):
    """
    Selecciona la componente con mayor relación de potencia espectral en rango fetal (1.8 - 3.0 Hz).
    """
    mejores_scores = []
    
    for ch in range(fuentes_ica.shape[1]):
        comp = fuentes_ica[:, ch]
        freqs, psd = signal.welch(comp, fs=fs, nperseg=int(fs*2))
        
        # Banda fetal: 1.8 a 3.0 Hz (~110-180 bpm)
        idx_fetal = np.logical_and(freqs >= 1.8, freqs <= 3.0)
        # Banda materna: 1.0 a 1.6 Hz (~60-96 bpm)
        idx_materno = np.logical_and(freqs >= 1.0, freqs <= 1.6)
        
        potencia_fetal = np.sum(psd[idx_fetal])
        potencia_materna = np.sum(psd[idx_materno]) + 1e-8
        
        score = potencia_fetal / potencia_materna
        mejores_scores.append(score)

    canal_fetal_idx = int(np.argmax(mejores_scores))
    return canal_fetal_idx, fuentes_ica[:, canal_fetal_idx]


def visualizar_proceso_preprocesamiento(sig_cruda, sig_filtrada, fuentes_ica, idx_fetal, fs=1000, seg=5):
    n_muestras = int(seg * fs)
    t = np.linspace(0, seg, n_muestras)

    fig, axes = plt.subplots(4, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("Etapa 1: Señales Filtradas vs Componentes FastICA", fontsize=13)

    for i in range(4):
        axes[i, 0].plot(t, sig_filtrada[:n_muestras, i], color='tab:blue', lw=0.8)
        axes[i, 0].set_ylabel(f"Filt Ch {i+1}")
        axes[i, 0].grid(True, linestyle='--', alpha=0.5)

        color = 'tab:red' if i == idx_fetal else 'tab:gray'
        label_fetal = " (Candidato Fetal)" if i == idx_fetal else ""
        axes[i, 1].plot(t, fuentes_ica[:n_muestras, i], color=color, lw=0.8)
        axes[i, 1].set_ylabel(f"ICA {i+1}{label_fetal}")
        axes[i, 1].grid(True, linestyle='--', alpha=0.5)

    axes[-1, 0].set_xlabel("Tiempo (s)")
    axes[-1, 1].set_xlabel("Tiempo (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Cargando registro a01 de CinC 2013...")
    sig_cruda, fs, fqrs_ann, ch_names = stream_cinc2013("a01")

    print("1. Aplicando filtros Pasa-banda y Notch (con limpieza de NaNs)...")
    sig_filt = aplicar_filtros(sig_cruda, fs)

    print("2. Ejecutando Separación Ciega de Fuentes (FastICA)...")
    fuentes_ica = extraer_componentes_ica(sig_filt)

    print("3. Seleccionando componente con predominancia fetal...")
    idx_fetal, canal_fetal = seleccionar_canal_fetal(fuentes_ica, fs)
    print(f"-> Componente seleccionada como fECG: ICA {idx_fetal + 1}")

    visualizar_proceso_preprocesamiento(sig_cruda, sig_filt, fuentes_ica, idx_fetal, fs, seg=5)