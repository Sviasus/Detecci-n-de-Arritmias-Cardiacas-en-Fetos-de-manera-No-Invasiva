import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_streamer import stream_nifeadb
from fqrs_detector import extraer_fqrs_optimo
from feature_extraction import extraer_vector_caracteristicas_completo

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "dataset_features.csv")


def procesar_registro(sig_4ch, fs):
    """
    Pipeline de preprocesamiento, detección de picos fQRS y extracción de fHRV.
    """
    try:
        picos = extraer_fqrs_optimo(sig_4ch, fs)
        if len(picos) < 10:
            return None
        return extraer_vector_caracteristicas_completo(picos, fs)
    except Exception:
        return None


def guardar_fila_csv(fila_dict):
    """
    Guarda progresivamente cada fila en disco para evitar pérdida de progreso.
    """
    df_fila = pd.DataFrame([fila_dict])
    if not os.path.exists(CSV_PATH):
        df_fila.to_csv(CSV_PATH, index=False)
    else:
        df_fila.to_csv(CSV_PATH, mode='a', header=False, index=False)


def obtener_registros_ya_procesados():
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            return set(df["Registro"].dropna().unique())
        except Exception:
            return set()
    return set()


def procesar_nifeadb():
    print("\n--- Procesando NIFEA DB (Validación Clínica Real: 26 casos) ---")
    registros_arr = [f"ARR_{i:02d}" for i in range(1, 13)]
    registros_nr = [f"NR_{i:02d}" for i in range(1, 15)]
    ya_procesados = obtener_registros_ya_procesados()

    for rec in tqdm(registros_arr, desc="NIFEA Arritmias (Clase 1)"):
        if rec in ya_procesados:
            continue
        try:
            sig, fs, _ = stream_nifeadb(rec)
            feats = procesar_registro(sig, fs)
            if feats is not None:
                feats["Registro"] = rec
                feats["Dataset"] = "NIFEA_DB"
                feats["Target"] = 1
                guardar_fila_csv(feats)
        except Exception as e:
            print(f"Error en {rec}: {e}")

    for rec in tqdm(registros_nr, desc="NIFEA Normales (Clase 0)"):
        if rec in ya_procesados:
            continue
        try:
            sig, fs, _ = stream_nifeadb(rec)
            feats = procesar_registro(sig, fs)
            if feats is not None:
                feats["Registro"] = rec
                feats["Dataset"] = "NIFEA_DB"
                feats["Target"] = 0
                guardar_fila_csv(feats)
        except Exception as e:
            print(f"Error en {rec}: {e}")


def procesar_fecgsyndb(limite_muestras=100):
    print(f"\n--- Procesando FECGSYNDB (Simulador: {limite_muestras} casos) ---")
    try:
        lista_remota = wfdb.get_record_list('fecgsyndb')
    except Exception as e:
        print(f"Error obteniendo lista remota de PhysioNet: {e}")
        return

    indices_cruz = [3, 12, 15, 27]
    ya_procesados = obtener_registros_ya_procesados()
    muestras = lista_remota[:limite_muestras]

    for item in tqdm(muestras, desc="FECGSYNDB"):
        sub_folder, rec_name = os.path.split(item)
        if rec_name in ya_procesados:
            continue
        try:
            pn_path = f"fecgsyndb/1.0.0/{sub_folder}"
            record = wfdb.rdrecord(rec_name, pn_dir=pn_path)
            sig_4ch = record.p_signal[:, indices_cruz]
            fs = record.fs

            feats = procesar_registro(sig_4ch, fs)
            if feats is not None:
                feats["Registro"] = rec_name
                feats["Dataset"] = "FECGSYNDB"
                feats["Target"] = 0 if "_c0" in rec_name else 1
                guardar_fila_csv(feats)
        except Exception:
            continue


if __name__ == "__main__":
    procesar_nifeadb()
    # 100 registros sintéticos balanceados son suficientes para pre-entrenamiento inicial
    procesar_fecgsyndb(limite_muestras=100)

    if os.path.exists(CSV_PATH):
        df_res = pd.read_csv(CSV_PATH)
        print("\n=======================================================")
        print(f"Dataset consolidado en: {CSV_PATH}")
        print(f"Total de registros listos: {len(df_res)}")
        print(f"Distribución de Clases (Target):\n{df_res['Target'].value_counts().to_string()}")
        print("=======================================================")