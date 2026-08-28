import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Directorio base para datos
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def cargar_cinc2013(record_name="a01"):
    """
    Descarga y lee registros de CinC Challenge 2013 (Set A).
    Retorna: signal (N, 4), fs (Hz), annotations (posiciones de fQRS), channel_names
    """
    out_dir = os.path.join(DATA_DIR, "cinc2013")
    os.makedirs(out_dir, exist_ok=True)
    
    # Descargar los archivos necesarios si no existen
    local_path = os.path.join(out_dir, "set-a", record_name)
    if not os.path.exists(f"{local_path}.hea"):
        wfdb.dl_files(
            db='challenge-2013',
            dl_dir=out_dir,
            files=[f'set-a/{record_name}.hea', f'set-a/{record_name}.dat', f'set-a/{record_name}.fqrs']
        )
    
    # Leer señal y anotaciones fQRS de referencia
    record = wfdb.rdrecord(local_path)
    try:
        ann = wfdb.rdann(local_path, extension='fqrs')
        fqrs_indices = ann.sample
    except Exception:
        fqrs_indices = None
        
    return record.p_signal, record.fs, fqrs_indices, record.sig_name


def cargar_fecgsyndb(sub="sub01", snr="snr00dB", case="c0"):
    """
    Descarga y homologa registros de FECGSYNDB a 4 canales (cruz abdominal).
    Retorna: signal (N, 4), fs (Hz), channel_names
    """
    out_dir = os.path.join(DATA_DIR, "fecgsyndb")
    record_rel_path = f"{sub}/{snr}/{sub}_{snr}_{case}"
    local_path = os.path.join(out_dir, record_rel_path)
    
    if not os.path.exists(f"{local_path}.hea"):
        wfdb.dl_files(
            db='fecgsyndb',
            dl_dir=out_dir,
            files=[f"{record_rel_path}.hea", f"{record_rel_path}.dat"]
        )
        
    record = wfdb.rdrecord(local_path)
    
    # Canales 4, 13, 16 y 28 (índices base-0: 3, 12, 15, 27)
    indices_cruz = [3, 12, 15, 27]
    signal_4ch = record.p_signal[:, indices_cruz]
    channel_names = [record.sig_name[i] for i in indices_cruz]
    
    return signal_4ch, record.fs, channel_names


def cargar_nifeadb(record_name="ARR_01"):
    """
    Descarga registros de NIFEA DB, descarta el canal torácico, 
    selecciona 4 canales abdominales y remuestrea a 1000 Hz si es necesario.
    Retorna: signal (N, 4), fs (Hz), channel_names
    """
    out_dir = os.path.join(DATA_DIR, "nifeadb")
    local_path = os.path.join(out_dir, record_name)
    
    if not os.path.exists(f"{local_path}.hea"):
        wfdb.dl_files(
            db='nifeadb',
            dl_dir=out_dir,
            files=[f"{record_name}.hea", f"{record_name}.dat"]
        )
        
    record = wfdb.rdrecord(local_path)
    
    # Filtrar únicamente canales abdominales (omitir 'Thorax')
    indices_abd = [
        i for i, name in enumerate(record.sig_name)
        if not ('tho' in name.lower() or 'chest' in name.lower())
    ]
    
    # Tomar 4 canales
    indices_4ch = indices_abd[:4]
    signal_4ch = record.p_signal[:, indices_4ch]
    
    target_fs = 1000
    if record.fs != target_fs:
        num_muestras = int(len(signal_4ch) * (target_fs / record.fs))
        signal_4ch = signal.resample(signal_4ch, num_muestras)
        fs_final = target_fs
    else:
        fs_final = record.fs
        
    channel_names = [record.sig_name[i] for i in indices_4ch]
    return signal_4ch, fs_final, channel_names


def visualizar_canales(sig, fs, channel_names, titulo, segundos=5):
    """
    Grafica los primeros N segundos de los 4 canales abdominales.
    """
    n_muestras = int(segundos * fs)
    tiempo = np.linspace(0, segundos, n_muestras)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"{titulo} (Primeros {segundos}s - Fs: {fs} Hz)", fontsize=12)
    
    for i in range(4):
        axes[i].plot(tiempo, sig[:n_muestras, i], color='tab:blue', linewidth=0.8)
        axes[i].set_ylabel(channel_names[i])
        axes[i].grid(True, linestyle='--', alpha=0.6)
        
    axes[-1].set_xlabel("Tiempo (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("--- 1. Probando CinC Challenge 2013 ---")
    sig_cinc, fs_cinc, fqrs_ann, ch_cinc = cargar_cinc2013("a01")
    print(f"Forma: {sig_cinc.shape}, Fs: {fs_cinc} Hz, Anotaciones fQRS: {len(fqrs_ann) if fqrs_ann is not None else 0}")
    visualizar_canales(sig_cinc, fs_cinc, ch_cinc, "CinC Challenge 2013 - a01")

    print("\n--- 2. Probando FECGSYNDB ---")
    sig_fecg, fs_fecg, ch_fecg = cargar_fecgsyndb("sub01", "snr00dB", "c0")
    print(f"Forma: {sig_fecg.shape}, Fs: {fs_fecg} Hz, Canales: {ch_fecg}")
    visualizar_canales(sig_fecg, fs_fecg, ch_fecg, "FECGSYNDB - sub01_snr00dB_c0 (Cruz Abdominal)")

    print("\n--- 3. Probando NIFEA DB ---")
    sig_nifea, fs_nifea, ch_nifea = cargar_nifeadb("ARR_01")
    print(f"Forma: {sig_nifea.shape}, Fs: {fs_nifea} Hz, Canales: {ch_nifea}")
    visualizar_canales(sig_nifea, fs_nifea, ch_nifea, "NIFEA DB - ARR_01 (4 Canales Abdominales)")