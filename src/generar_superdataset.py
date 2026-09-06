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

from fqrs_detector import extraer_fqrs_optimo
from feature_extraction import (
    calcular_tacograma_rr,
    extraer_features_temporales,
    extraer_features_frecuenciales,
    extraer_features_no_lineales,
    extraer_vector_caracteristicas_completo
)

DATA_DIR = PROJECT_ROOT / "data"
CINC_DIR = DATA_DIR / "cinc2013_real" / "set-a"
CSV_PATH = DATA_DIR / "dataset_features.csv"


def procesar_registro_cinc(rec_name):
    """
    Procesa un registro de CinC Challenge 2013 desde el disco local.
    """
    path_base = str(CINC_DIR / rec_name)
    try:
        record = wfdb.rdrecord(path_base)
        sig_4ch = record.p_signal
        fs = float(record.fs)

        muestras = []
        # 1. Registro completo de 60 segundos
        picos = extraer_fqrs_optimo(sig_4ch, fs)
        if len(picos) >= 15:
            feats = extraer_vector_caracteristicas_completo(picos, fs)
            feats["Registro"] = rec_name
            feats["Dataset"] = "CinC_2013"
            feats["Target"] = 0
            muestras.append(feats)

        # 2. Ventanas de 35 segundos con paso de 15 segundos
        w_pts = int(35.0 * fs)
        step_pts = int(15.0 * fs)
        for ini in range(0, len(sig_4ch) - w_pts + 1, step_pts):
            sub_sig = sig_4ch[ini:ini + w_pts]
            sub_picos = extraer_fqrs_optimo(sub_sig, fs)
            if len(sub_picos) >= 12:
                sub_feats = extraer_vector_caracteristicas_completo(sub_picos, fs)
                sub_feats["Registro"] = rec_name
                sub_feats["Dataset"] = "CinC_2013"
                sub_feats["Target"] = 0
                muestras.append(sub_feats)

        return muestras
    except Exception:
        return []


def aplicar_aumento_fisiologico(df_base, total_objetivo=1800):
    """
    Aumento de datos fisiológico (Data Augmentation):
    Simula variabilidad estocástica del tono autonómico fetal (+/- 1.5% a 2.5% de variación natural)
    para enriquecer ambas clases de forma clínicamente verosímil sin alterar la patología.
    """
    columnas_excluidas = ["Registro", "Dataset", "Target"]
    features = [c for c in df_base.columns if c not in columnas_excluidas]

    df_aumentado = [df_base.copy()]
    muestras_actuales = len(df_base)
    faltantes = max(0, total_objetivo - muestras_actuales)

    if faltantes <= 0:
        return df_base

    print(f"\nAplicando aumento de datos fisiológico para alcanzar ~{total_objetivo} registros...")

    # Generar variantes plausibles con perturbaciones normales del tono autonómico
    np.random.seed(42)
    indices_muestreo = np.random.choice(len(df_base), size=faltantes, replace=True)

    filas_nuevas = []
    for idx in indices_muestreo:
        fila_orig = df_base.iloc[idx].to_dict()
        fila_nueva = fila_orig.copy()

        # Perturbación sutil (+/- 1.5% a 3%) que emula variaciones circadianas/autonómicas fetales
        factor_bpm = 1.0 + np.random.normal(0, 0.018)
        factor_hrv = 1.0 + np.random.normal(0, 0.025)

        fila_nueva["BPM_mean"] = float(fila_orig["BPM_mean"] * factor_bpm)
        fila_nueva["SDNN"] = float(max(1.0, fila_orig["SDNN"] * factor_hrv))
        fila_nueva["RMSSD"] = float(max(1.0, fila_orig["RMSSD"] * factor_hrv))
        fila_nueva["SD1"] = float(max(0.5, fila_orig["SD1"] * factor_hrv))
        fila_nueva["SD2"] = float(max(1.0, fila_orig["SD2"] * factor_hrv))
        fila_nueva["SampEn"] = float(max(0.01, fila_orig["SampEn"] * (1.0 + np.random.normal(0, 0.02))))
        fila_nueva["VLF"] = float(max(0.0, fila_orig["VLF"] * (factor_hrv ** 2)))
        fila_nueva["LF"] = float(max(0.0, fila_orig["LF"] * (factor_hrv ** 2)))
        fila_nueva["HF"] = float(max(0.0, fila_orig["HF"] * (factor_hrv ** 2)))
        fila_nueva["Registro"] = f"{fila_orig['Registro']}_aug"

        filas_nuevas.append(fila_nueva)

    df_aug = pd.DataFrame(filas_nuevas)
    df_resultado = pd.concat([df_base, df_aug], ignore_index=True)

    # Eliminar duplicados exactos
    df_resultado = df_resultado.drop_duplicates(subset=features).reset_index(drop=True)
    return df_resultado


def construir_superdataset(total_objetivo=1800):
    print("=" * 75)
    print(f"      CONSTRUCCIÓN DEL SUPERDATASET CLÍNICO EXTENDIDO (~{total_objetivo} MUESTRAS)      ")
    print("=" * 75)

    # 1. Cargar las muestras ya procesadas de NIFEA DB
    if CSV_PATH.exists():
        df_nifea = pd.read_csv(CSV_PATH)
        print(f"Muestras base de NIFEA DB disponibles: {len(df_nifea)}")
    else:
        df_nifea = pd.DataFrame()

    # 2. Procesar los registros locales de CinC Challenge 2013
    print("\nProcesando los 75 registros reales de CinC Challenge 2013 (locales)...")
    filas_cinc = []
    for i in tqdm(range(1, 76), desc="CinC 2013 Set-A"):
        rec_name = f"a{i:02d}"
        path_dat = CINC_DIR / f"{rec_name}.dat"
        if path_dat.exists():
            muestras = procesar_registro_cinc(rec_name)
            filas_cinc.extend(muestras)

    df_cinc = pd.DataFrame(filas_cinc)
    print(f"Muestras reales extraídas de CinC 2013: {len(df_cinc)}")

    # 3. Consolidar registros reales
    df_combinado = pd.concat([df_nifea, df_cinc], ignore_index=True)
    print(f"Muestras reales combinadas: {len(df_combinado)}")
    print(f"Distribución antes del aumento:\n{df_combinado['Target'].value_counts().to_string()}")

    # 4. Aumento Fisiológico para alcanzar el volumen objetivo masivo
    df_final = aplicar_aumento_fisiologico(df_combinado, total_objetivo=total_objetivo)

    # 5. Guardar el nuevo dataset
    df_final.to_csv(CSV_PATH, index=False)
    print("\n" + "=" * 75)
    print(f"[OK] SUPERDATASET consolidado exitosamente en: {CSV_PATH}")
    print(f"Total de registros finales : {len(df_final)}")
    print(f"Distribución de Clases     :\n{df_final['Target'].value_counts().to_string()}")
    print("=" * 75)
    return df_final


if __name__ == "__main__":
    construir_superdataset(total_objetivo=1850)
