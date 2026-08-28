import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# -------------------------------------------------------------------------
# 1. CinC Challenge 2013 (Set A)
# -------------------------------------------------------------------------
def stream_cinc2013(record_name="a01"):
    """
    Lee en memoria directamente desde CinC 2013 (Set A).
    """
    record = wfdb.rdrecord(record_name, pn_dir="challenge-2013/1.0.0/set-a")
    try:
        ann = wfdb.rdann(record_name, extension='fqrs', pn_dir="challenge-2013/1.0.0/set-a")
        fqrs_indices = ann.sample
    except Exception:
        fqrs_indices = None
        
    return record.p_signal, record.fs, fqrs_indices, record.sig_name


# -------------------------------------------------------------------------
# 2. FECGSYNDB (Simulador)
# -------------------------------------------------------------------------
def stream_fecgsyndb(index=0):
    """
    Obtiene la ruta oficial de la lista de PhysioNet y extrae los 4 canales de la cruz abdominal.
    """
    # Consulta la lista real y oficial de registros remotos
    record_list = wfdb.get_record_list('fecgsyndb')
    full_path = record_list[index]  # Ej: 'sub01/snr00dB/sub01_snr00dB_c0'
    
    # Separar subcarpeta del nombre del archivo
    sub_folder, rec_name = os.path.split(full_path)
    pn_path = f"fecgsyndb/1.0.0/{sub_folder}"
    
    record = wfdb.rdrecord(rec_name, pn_dir=pn_path)
    
    # Extraer canales 4, 13, 16 y 28 (índices Python base 0: 3, 12, 15, 27)
    indices_cruz = [3, 12, 15, 27]
    signal_4ch = record.p_signal[:, indices_cruz]
    channel_names = [record.sig_name[i] for i in indices_cruz]
    
    return signal_4ch, record.fs, channel_names, full_path


# -------------------------------------------------------------------------
# 3. NIFEA DB (Clínica Real)
# -------------------------------------------------------------------------
def stream_nifeadb(record_name="ARR_01"):
    """
    Lee registros clínicos reales ('ARR_01' a 'ARR_12', 'NR_01' a 'NR_14'),
    descarta canal torácico, toma 4 abdominales y remuestrea a 1000 Hz si viene a 500 Hz.
    """
    record = wfdb.rdrecord(record_name, pn_dir="nifeadb/1.0.0")
    
    # Excluir canal torácico
    indices_abd = [
        i for i, name in enumerate(record.sig_name)
        if not ('tho' in name.lower() or 'chest' in name.lower())
    ]
    
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


# -------------------------------------------------------------------------
# Visualizador
# -------------------------------------------------------------------------
def visualizar_canales(sig, fs, channel_names, titulo, segundos=5):
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
    print("1. Consultando CinC 2013 (a01)...")
    sig_cinc, fs_cinc, ann_cinc, ch_cinc = stream_cinc2013("a01")
    print(f"-> CinC 2013: {sig_cinc.shape} | Fs: {fs_cinc} Hz | Picos fQRS: {len(ann_cinc) if ann_cinc is not None else 0}")
    visualizar_canales(sig_cinc, fs_cinc, ch_cinc, "Streaming: CinC 2013 (a01)")

    print("\n2. Consultando FECGSYNDB vía lista remota...")
    sig_fecg, fs_fecg, ch_fecg, nombre_fecg = stream_fecgsyndb(index=0)
    print(f"-> FECGSYNDB ({nombre_fecg}): {sig_fecg.shape} | Fs: {fs_fecg} Hz | Canales: {ch_fecg}")
    visualizar_canales(sig_fecg, fs_fecg, ch_fecg, f"Streaming: FECGSYNDB ({nombre_fecg})")

    print("\n3. Consultando NIFEA DB (ARR_01)...")
    sig_nifea, fs_nifea, ch_nifea = stream_nifeadb("ARR_01")
    print(f"-> NIFEA DB: {sig_nifea.shape} | Fs: {fs_nifea} Hz | Canales: {ch_nifea}")
    visualizar_canales(sig_nifea, fs_nifea, ch_nifea, "Streaming: NIFEA DB (ARR_01)")