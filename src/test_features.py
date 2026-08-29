import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Resolución de ruta local para importar src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data_streamer import stream_nifeadb
from fqrs_detector import extraer_fqrs_optimo
from feature_extraction import (
    calcular_tacograma_rr,
    extraer_vector_caracteristicas_completo
)


def ejecutar_test_comparativo():
    print("===============================================================")
    print("      TEST DE VERIFICACIÓN DE CARACTERÍSTICAS fHRV             ")
    print("===============================================================")

    # 1. Cargar Muestra Normal (NR_01)
    print("\n[1/2] Extrayendo tacograma y fHRV de caso NORMAL (NR_01)...")
    sig_norm, fs_norm, _ = stream_nifeadb("NR_01")
    picos_norm = extraer_fqrs_optimo(sig_norm, fs_norm)
    rr_norm = calcular_tacograma_rr(picos_norm, fs_norm)
    feats_norm = extraer_vector_caracteristicas_completo(picos_norm, fs_norm)

    # 2. Cargar Muestra con Arritmia (ARR_01)
    print("[2/2] Extrayendo tacograma y fHRV de caso ARRITMIA (ARR_01)...")
    sig_arr, fs_arr, _ = stream_nifeadb("ARR_01")
    picos_arr = extraer_fqrs_optimo(sig_arr, fs_arr)
    rr_arr = calcular_tacograma_rr(picos_arr, fs_arr)
    feats_arr = extraer_vector_caracteristicas_completo(picos_arr, fs_arr)

    # 3. Mostrar Tabla Comparativa de Descriptores
    df_comp = pd.DataFrame([feats_norm, feats_arr], index=["Normal (NR_01)", "Arritmia (ARR_01)"]).T
    print("\n--- TABLA COMPARATIVA DE FEATURES EXTRAÍDAS ---")
    print(df_comp.round(3).to_string())

    # 4. Visualización Diagnóstica: Tacograma RR y Gráfico de Poincaré
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Verificación de Tacograma RR y Diagrama de Poincaré (Normal vs Arritmia)", fontsize=13)

    # Tacograma Normal
    axes[0, 0].plot(rr_norm, color='tab:blue', lw=1.2, marker='.', markersize=4)
    axes[0, 0].set_title(f"Tacograma RR Normal (BPM prom: {feats_norm['BPM_mean']:.1f} | SDNN: {feats_norm['SDNN']:.1f} ms)")
    axes[0, 0].set_ylabel("Intervalo RR (ms)")
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # Tacograma Arritmia
    axes[0, 1].plot(rr_arr, color='tab:red', lw=1.2, marker='.', markersize=4)
    axes[0, 1].set_title(f"Tacograma RR Arritmia (BPM prom: {feats_arr['BPM_mean']:.1f} | SDNN: {feats_arr['SDNN']:.1f} ms)")
    axes[0, 1].set_ylabel("Intervalo RR (ms)")
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # Poincaré Normal (RR_i vs RR_i+1)
    axes[1, 0].scatter(rr_norm[:-1], rr_norm[1:], color='tab:blue', alpha=0.7, edgecolors='none')
    axes[1, 0].set_title(f"Poincaré Normal (SD1: {feats_norm['SD1']:.1f} | SD2: {feats_norm['SD2']:.1f})")
    axes[1, 0].set_xlabel("RR(n) [ms]")
    axes[1, 0].set_ylabel("RR(n+1) [ms]")
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    # Poincaré Arritmia (RR_i vs RR_i+1)
    axes[1, 1].scatter(rr_arr[:-1], rr_arr[1:], color='tab:red', alpha=0.7, edgecolors='none')
    axes[1, 1].set_title(f"Poincaré Arritmia (SD1: {feats_arr['SD1']:.1f} | SD2: {feats_arr['SD2']:.1f})")
    axes[1, 1].set_xlabel("RR(n) [ms]")
    axes[1, 1].set_ylabel("RR(n+1) [ms]")
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ejecutar_test_comparativo()