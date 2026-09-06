from concurrent.futures import as_completed, ThreadPoolExecutor
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
  sys.path.append(CURRENT_DIR)

from data_streamer import stream_nifeadb
from feature_extraction import extraer_vector_caracteristicas_completo
from fqrs_detector import extraer_fqrs_optimo

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "dataset_features_full.csv")


def procesar_registro_individual(sig_4ch, fs):
  try:
    picos = extraer_fqrs_optimo(sig_4ch, fs)
    if len(picos) < 10:
      return None
    return extraer_vector_caracteristicas_completo(picos, fs)
  except Exception:
    return None


def guardar_lote_csv(lista_dicts):
  if not lista_dicts:
    return
  df_lote = pd.DataFrame(lista_dicts)
  if not os.path.exists(CSV_PATH):
    df_lote.to_csv(CSV_PATH, index=False)
  else:
    df_lote.to_csv(CSV_PATH, mode="a", header=False, index=False)


def obtener_registros_ya_procesados():
  if os.path.exists(CSV_PATH):
    try:
      df = pd.read_csv(CSV_PATH)
      return set(df["Registro"].dropna().unique())
    except Exception:
      return set()
  return set()


def procesar_un_caso_fecgsyndb(item):
  sub_folder, rec_name = os.path.split(item)
  indices_cruz = [3, 12, 15, 27]
  try:
    pn_path = f"fecgsyndb/1.0.0/{sub_folder}"
    record = wfdb.rdrecord(rec_name, pn_dir=pn_path)
    sig_4ch = record.p_signal[:, indices_cruz]
    fs = record.fs

    feats = procesar_registro_individual(sig_4ch, fs)
    if feats is not None:
      feats["Registro"] = rec_name
      feats["Dataset"] = "FECGSYNDB"
      feats["Target"] = 0 if "_c0" in rec_name else 1
      return feats
  except Exception:
    pass
  return None


def ejecutar_extraccion_masiva(max_workers=6, total_objetivo=1500):
    # 1. NIFEA DB (26 Casos con Barra de Progreso y Contador)
    registros_nifea = [f"ARR_{i:02d}" for i in range(1, 13)] + [
        f"NR_{i:02d}" for i in range(1, 15)
    ]
    ya_procesados = obtener_registros_ya_procesados()
    pendientes_nifea = [r for r in registros_nifea if r not in ya_procesados]

    print("\n--- Procesando NIFEA DB (Validación Clínica Real: 26 Casos) ---")
    if pendientes_nifea:
        for rec in tqdm(
            pendientes_nifea,
            desc="NIFEA DB",
            unit="paciente",
            dynamic_ncols=True,
        ):
            try:
                sig, fs, _ = stream_nifeadb(rec)
                feats = procesar_registro_individual(sig, fs)
                if feats is not None:
                    feats["Registro"] = rec
                    feats["Dataset"] = "NIFEA_DB"
                    feats["Target"] = 1 if "ARR" in rec else 0
                    guardar_lote_csv([feats])
            except Exception as e:
                print(f"\n[Aviso] Error procesando {rec}: {e}")
    else:
        print("Todos los registros de NIFEA DB ya se encuentran en el archivo CSV.")

    # 2. FECGSYNDB (Muestreo Representativo y Balanceado de ~1500 Casos)
    print(f"\n--- Procesando FECGSYNDB Representativo (~{total_objetivo} casos en paralelo con {max_workers} hilos) ---")
    try:
        lista_completa = wfdb.get_record_list("fecgsyndb")
    except Exception as e:
        print(f"Error consultando la lista remota de PhysioNet: {e}")
        return

    # Muestreo estratificado uniforme a lo largo de las 7,000 combinaciones
    paso = max(1, len(lista_completa) // total_objetivo)
    lista_seleccionada = lista_completa[::paso][:total_objetivo]

    ya_procesados = obtener_registros_ya_procesados()
    casos_pendientes = [
        c for c in lista_seleccionada if os.path.split(c)[1] not in ya_procesados
    ]
    print(
        f"Casos representativos a procesar: {len(casos_pendientes)} de {len(lista_seleccionada)}"
    )

    buffer = []
    batch_size = 15

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {
            executor.submit(procesar_un_caso_fecgsyndb, caso): caso
            for caso in casos_pendientes
        }

        for future in tqdm(
            as_completed(futuros),
            total=len(futuros),
            desc="FECGSYNDB Multihilo",
            unit="registro",
            dynamic_ncols=True,
        ):
            res = future.result()
            if res is not None:
                buffer.append(res)
                if len(buffer) >= batch_size:
                    guardar_lote_csv(buffer)
                    buffer = []

    if buffer:
        guardar_lote_csv(buffer)

    df_final = pd.read_csv(CSV_PATH)
    print("\n=======================================================")
    print("Extracción masiva finalizada con éxito.")
    print(f"Archivo generado: {CSV_PATH}")
    print(f"Total de registros consolidados: {len(df_final)}")
    print(
        f"Distribución de Clases:\n{df_final['Target'].value_counts().to_string()}"
    )
    print("=======================================================")


if __name__ == "__main__":
  ejecutar_extraccion_masiva(max_workers=6)