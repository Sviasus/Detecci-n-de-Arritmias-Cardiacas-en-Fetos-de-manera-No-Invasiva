# ============================================================
# VISUALIZACIÓN DE LA CARACTERIZACIÓN DE SEÑALES
# Detector de Arritmias Cardíacas Fetales
#
# Utiliza las características generadas por:
#     feature_extraction.py
#
# Genera:
#   1. Pipeline fQRS -> RR -> características
#   2. Tacograma RR
#   3. Distribución de características
#   4. Violin plots
#   5. Boxplots
#   6. Matriz de correlación
# ============================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"

CSV_PATH = DATA_DIR / "dataset_features.csv"

RESULTADOS = PROJECT_ROOT / "resultados_caracterizacion"

RESULTADOS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMPORTAR FUNCIONES REALES DEL PROYECTO
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from feature_extraction import (
    calcular_tacograma_rr,
    extraer_features_temporales,
    extraer_features_frecuenciales,
    extraer_features_no_lineales,
    extraer_vector_caracteristicas_completo
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Cantidad máxima de muestras que se mostrarán en los gráficos
MAX_MUESTRAS = 100

# Características utilizadas
FEATURES_TEMPORALES = [
    "BPM_mean",
    "SDNN",
    "RMSSD",
    "pNN50"
]

FEATURES_FRECUENCIALES = [
    "VLF",
    "LF",
    "HF",
    "LF_HF_ratio"
]

FEATURES_NO_LINEALES = [
    "SD1",
    "SD2",
    "SD1_SD2_ratio",
    "SampEn",
    "DFA_alpha1"
]

FEATURES = (
    FEATURES_TEMPORALES
    + FEATURES_FRECUENCIALES
    + FEATURES_NO_LINEALES
)


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("VISUALIZACIÓN DE LA CARACTERIZACIÓN")
print("=" * 70)

print(f"\nProyecto:")
print(PROJECT_ROOT)

print(f"\nDataset:")
print(CSV_PATH)


# ============================================================
# COMPROBAR DATASET
# ============================================================

if not CSV_PATH.exists():

    print("\n❌ No se encontró el dataset:")
    print(CSV_PATH)

    print(
        "\nPrimero ejecuta build_dataset.py "
        "para generar dataset_features.csv."
    )

    raise SystemExit


# ============================================================
# CARGAR DATASET
# ============================================================

print("\nCargando dataset...")

df = pd.read_csv(
    CSV_PATH
)

print("✓ Dataset cargado.")

print(
    f"\nNúmero de muestras: {len(df)}"
)

print(
    f"Número de columnas: {len(df.columns)}"
)

print("\nColumnas:")

print(
    list(df.columns)
)


# ============================================================
# COMPROBAR CARACTERÍSTICAS
# ============================================================

features_disponibles = [
    f for f in FEATURES
    if f in df.columns
]

features_faltantes = [
    f for f in FEATURES
    if f not in df.columns
]

print("\nCaracterísticas encontradas:")

for f in features_disponibles:
    print(f"  ✓ {f}")


if features_faltantes:

    print("\n⚠ Características faltantes:")

    for f in features_faltantes:
        print(f"  - {f}")


# ============================================================
# LIMPIEZA NUMÉRICA
# ============================================================

df_features = df[
    features_disponibles
].copy()

df_features = df_features.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# ============================================================
# PARTE 1
# EJEMPLO DEL PROCESAMIENTO DE UNA MUESTRA
# ============================================================
# ============================================================

print("\n")
print("=" * 70)
print("PARTE 1 — PROCESAMIENTO DE UNA MUESTRA")
print("=" * 70)


# ------------------------------------------------------------
# Seleccionar una muestra del dataset
# ------------------------------------------------------------

fila = df.iloc[0]

print(
    f"\nMuestra seleccionada:"
)

if "Registro" in df.columns:
    print(
        f"Registro: {fila['Registro']}"
    )

if "Dataset" in df.columns:
    print(
        f"Dataset: {fila['Dataset']}"
    )

if "Target" in df.columns:
    print(
        f"Target: {fila['Target']}"
    )


# ------------------------------------------------------------
# Mostrar las características extraídas
# ------------------------------------------------------------

print("\nCaracterísticas extraídas:")

for feature in features_disponibles:

    valor = fila[feature]

    print(
        f"  {feature:<20} = {valor:.4f}"
    )


# ============================================================
# 2. VISUALIZACIÓN DE LAS FAMILIAS DE CARACTERÍSTICAS
# ============================================================

# ------------------------------------------------------------
# Valores de la muestra
# ------------------------------------------------------------

valores_temp = [
    fila[f]
    for f in FEATURES_TEMPORALES
    if f in df.columns
]

nombres_temp = [
    f
    for f in FEATURES_TEMPORALES
    if f in df.columns
]


valores_freq = [
    fila[f]
    for f in FEATURES_FRECUENCIALES
    if f in df.columns
]

nombres_freq = [
    f
    for f in FEATURES_FRECUENCIALES
    if f in df.columns
]


valores_nl = [
    fila[f]
    for f in FEATURES_NO_LINEALES
    if f in df.columns
]

nombres_nl = [
    f
    for f in FEATURES_NO_LINEALES
    if f in df.columns
]


# ============================================================
# GRÁFICA — CARACTERÍSTICAS TEMPORALES
# ============================================================

if nombres_temp:

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        nombres_temp,
        valores_temp
    )

    plt.title(
        "Características en el dominio temporal"
    )

    plt.ylabel(
        "Valor"
    )

    plt.xticks(
        rotation=25
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTADOS /
        "01_caracteristicas_temporales.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# GRÁFICA — CARACTERÍSTICAS FRECUENCIALES
# ============================================================

if nombres_freq:

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        nombres_freq,
        valores_freq
    )

    plt.title(
        "Características en el dominio frecuencial"
    )

    plt.ylabel(
        "Potencia / razón"
    )

    plt.xticks(
        rotation=25
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTADOS /
        "02_caracteristicas_frecuenciales.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# GRÁFICA — CARACTERÍSTICAS NO LINEALES
# ============================================================

if nombres_nl:

    plt.figure(
        figsize=(11, 5)
    )

    plt.bar(
        nombres_nl,
        valores_nl
    )

    plt.title(
        "Características no lineales"
    )

    plt.ylabel(
        "Valor"
    )

    plt.xticks(
        rotation=25
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTADOS /
        "03_caracteristicas_no_lineales.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# ============================================================
# PARTE 2
# DISTRIBUCIÓN DE LAS CARACTERÍSTICAS
# ============================================================
# ============================================================

print("\n")
print("=" * 70)
print("PARTE 2 — DISTRIBUCIÓN DE CARACTERÍSTICAS")
print("=" * 70)


# ------------------------------------------------------------
# Seleccionar muestras
# ------------------------------------------------------------

df_plot = df.copy()

if len(df_plot) > MAX_MUESTRAS:

    df_plot = df_plot.head(
        MAX_MUESTRAS
    )

print(
    f"\nMuestras utilizadas para visualización: "
    f"{len(df_plot)}"
)


# ============================================================
# VIOLIN PLOT
# ============================================================

print("\nGenerando violin plots...")


# ------------------------------------------------------------
# Función para crear violin plots
# ------------------------------------------------------------

def crear_violin(
    columnas,
    titulo,
    nombre_archivo
):

    columnas = [
        c
        for c in columnas
        if c in df_plot.columns
    ]

    if not columnas:
        return

    datos = []

    etiquetas = []

    for columna in columnas:

        valores = pd.to_numeric(
            df_plot[columna],
            errors="coerce"
        ).dropna()

        if len(valores) > 0:

            datos.append(
                valores.values
            )

            etiquetas.append(
                columna
            )


    if not datos:
        return


    plt.figure(
        figsize=(13, 6)
    )

    plt.violinplot(
        datos,
        showmeans=True,
        showmedians=True
    )

    plt.xticks(
        range(1, len(etiquetas) + 1),
        etiquetas,
        rotation=25
    )

    plt.title(
        titulo,
        fontsize=15,
        fontweight="bold"
    )

    plt.ylabel(
        "Valor"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTADOS /
        nombre_archivo,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ------------------------------------------------------------
# Violin temporal
# ------------------------------------------------------------

crear_violin(
    FEATURES_TEMPORALES,
    "Distribución de características temporales",
    "04_violin_temporales.png"
)


# ------------------------------------------------------------
# Violin frecuencial
# ------------------------------------------------------------

crear_violin(
    FEATURES_FRECUENCIALES,
    "Distribución de características frecuenciales",
    "05_violin_frecuenciales.png"
)


# ------------------------------------------------------------
# Violin no lineal
# ------------------------------------------------------------

crear_violin(
    FEATURES_NO_LINEALES,
    "Distribución de características no lineales",
    "06_violin_no_lineales.png"
)


# ============================================================
# ============================================================
# PARTE 3
# BOXPLOTS
# ============================================================
# ============================================================

print("\nGenerando boxplots...")


def crear_boxplot(
    columnas,
    titulo,
    nombre_archivo
):

    columnas = [
        c
        for c in columnas
        if c in df_plot.columns
    ]

    if not columnas:
        return


    datos = []

    etiquetas = []


    for columna in columnas:

        valores = pd.to_numeric(
            df_plot[columna],
            errors="coerce"
        ).dropna()

        if len(valores) > 0:

            datos.append(
                valores.values
            )

            etiquetas.append(
                columna
            )


    if not datos:
        return


    plt.figure(
        figsize=(13, 6)
    )

    plt.boxplot(
        datos,
        tick_labels=etiquetas,
        showmeans=True
    )

    plt.title(
        titulo,
        fontsize=15,
        fontweight="bold"
    )

    plt.ylabel(
        "Valor"
    )

    plt.xticks(
        rotation=25
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTADOS /
        nombre_archivo,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ------------------------------------------------------------
# Boxplot temporal
# ------------------------------------------------------------

crear_boxplot(
    FEATURES_TEMPORALES,
    "Distribución — características temporales",
    "07_boxplot_temporales.png"
)


# ------------------------------------------------------------
# Boxplot frecuencial
# ------------------------------------------------------------

crear_boxplot(
    FEATURES_FRECUENCIALES,
    "Distribución — características frecuenciales",
    "08_boxplot_frecuenciales.png"
)


# ------------------------------------------------------------
# Boxplot no lineal
# ------------------------------------------------------------

crear_boxplot(
    FEATURES_NO_LINEALES,
    "Distribución — características no lineales",
    "09_boxplot_no_lineales.png"
)


# ============================================================
# ============================================================
# PARTE 4
# MATRIZ DE CORRELACIÓN
# ============================================================
# ============================================================

print("\n")
print("=" * 70)
print("PARTE 4 — MATRIZ DE CORRELACIÓN")
print("=" * 70)


df_corr = df[
    features_disponibles
].copy()

df_corr = df_corr.apply(
    pd.to_numeric,
    errors="coerce"
)


corr = df_corr.corr(
    method="pearson"
)


print("\nMatriz de correlación:")

print(
    corr.round(2)
)


# ============================================================
# GRÁFICA DE CORRELACIÓN
# ============================================================

plt.figure(
    figsize=(13, 11)
)

plt.imshow(
    corr,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar(
    label="Correlación de Pearson"
)


plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=70
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)


plt.title(
    "Matriz de correlación de las características",
    fontsize=16,
    fontweight="bold"
)


# ------------------------------------------------------------
# Escribir valores dentro de la matriz
# ------------------------------------------------------------

for i in range(
    len(corr.columns)
):

    for j in range(
        len(corr.columns)
    ):

        valor = corr.iloc[i, j]

        if not np.isnan(valor):

            plt.text(
                j,
                i,
                f"{valor:.2f}",
                ha="center",
                va="center",
                fontsize=7
            )


plt.tight_layout()

plt.savefig(
    RESULTADOS /
    "10_matriz_correlacion.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ============================================================
# PARTE 5
# RESUMEN ESTADÍSTICO
# ============================================================
# ============================================================

print("\n")
print("=" * 70)
print("PARTE 5 — RESUMEN ESTADÍSTICO")
print("=" * 70)


resumen = df[
    features_disponibles
].describe().T


print(
    resumen[
        [
            "count",
            "mean",
            "std",
            "min",
            "50%",
            "max"
        ]
    ].round(3)
)


# Guardar resumen

resumen.to_csv(
    RESULTADOS /
    "resumen_estadistico_caracteristicas.csv"
)


# ============================================================
# GUARDAR MATRIZ
# ============================================================

corr.to_csv(
    RESULTADOS /
    "matriz_correlacion.csv"
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("✓ CARACTERIZACIÓN VISUALIZADA")
print("=" * 70)

print(
    "\nResultados guardados en:"
)

print(
    RESULTADOS
)

print("\nArchivos generados:")

for archivo in sorted(
    RESULTADOS.glob("*")
):

    print(
        f"  ✓ {archivo.name}"
    )