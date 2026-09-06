import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from data_streamer import stream_nifeadb
from fqrs_detector import extraer_fqrs_optimo
from feature_extraction import extraer_vector_caracteristicas_completo

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "dataset_features.csv"


def procesar_registro(sig_4ch, fs):
    """
    Pipeline de la Fase 2: preprocesamiento, cancelación materna mQRS,
    separación FastICA y extracción de fHRV con bandas fetales reales.
    """
    try:
        picos = extraer_fqrs_optimo(sig_4ch, fs)
        if len(picos) < 10:
            return None
        return extraer_vector_caracteristicas_completo(picos, fs)
    except Exception:
        return None


def extraer_segmentos_paciente(sig_4ch, fs, duracion_ventana_s=90.0, paso_s=45.0):
    """
    Segmenta un registro largo en ventanas clínicas para enriquecer el dataset
    preservando el identificador de paciente para GroupKFold.
    """
    n_samples = len(sig_4ch)
    pts_ventana = int(duracion_ventana_s * fs)
    pts_paso = int(paso_s * fs)
    vectores = []

    # 1. Procesar el registro completo primero
    v_completo = procesar_registro(sig_4ch, fs)
    if v_completo is not None:
        vectores.append(v_completo)

    # 2. Si la señal es lo bastante extensa, extraer ventanas deslizantes
    if n_samples >= pts_ventana + pts_paso:
        for inicio in range(0, n_samples - pts_ventana + 1, pts_paso):
            segmento = sig_4ch[inicio:inicio + pts_ventana]
            v_seg = procesar_registro(segmento, fs)
            if v_seg is not None and v_seg.get("Num_Latidos_Validos", 0) >= 15:
                vectores.append(v_seg)

    return vectores


def construir_dataset_nifeadb(reset=False):
    """
    Procesa los 26 pacientes clínicos reales de NIFEA DB (12 con arritmia y 14 normales).
    """
    print("\n--- Procesando Pacientes Clínicos Reales de NIFEA DB (Fase 3) ---")
    registros_arr = [f"ARR_{i:02d}" for i in range(1, 13)]
    registros_nr = [f"NR_{i:02d}" for i in range(1, 15)]

    filas_acumuladas = []

    # 1. Casos de Arritmia Clínica (Clase 1)
    for rec in tqdm(registros_arr, desc="NIFEA Arritmias (Clase 1)"):
        try:
            sig, fs, _ = stream_nifeadb(rec)
            segmentos = extraer_segmentos_paciente(sig, fs, duracion_ventana_s=90.0, paso_s=45.0)
            for v in segmentos:
                v["Registro"] = rec
                v["Dataset"] = "NIFEA_DB"
                v["Target"] = 1
                filas_acumuladas.append(v)
        except Exception as e:
            print(f"Aviso: error en {rec}: {e}")

    # 2. Casos Normales de Control (Clase 0)
    for rec in tqdm(registros_nr, desc="NIFEA Controles Sanos (Clase 0)"):
        try:
            sig, fs, _ = stream_nifeadb(rec)
            segmentos = extraer_segmentos_paciente(sig, fs, duracion_ventana_s=90.0, paso_s=45.0)
            for v in segmentos:
                v["Registro"] = rec
                v["Dataset"] = "NIFEA_DB"
                v["Target"] = 0
                filas_acumuladas.append(v)
        except Exception as e:
            print(f"Aviso: error en {rec}: {e}")

    df = pd.DataFrame(filas_acumuladas)

    # Eliminar posibles filas duplicadas exactas en las características numéricas
    cols_meta = ["Registro", "Dataset", "Target"]
    cols_feat = [c for c in df.columns if c not in cols_meta]
    df = df.drop_duplicates(subset=cols_feat).reset_index(drop=True)

    df.to_csv(CSV_PATH, index=False)
    print(f"\n[OK] Dataset generado exitosamente en: {CSV_PATH}")
    print(f"Total de muestras clínicas: {len(df)}")

    print(f"Pacientes únicos representados: {df['Registro'].nunique()}")
    print(f"Distribución de Clases:\n{df['Target'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    construir_dataset_nifeadb(reset=True)