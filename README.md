# 🩺 Detección No Invasiva de Arritmias Cardíacas Fetales (ni-fECG)

Sistema integral de ingeniería biomédica y ciencia de datos diseñado para la adquisición, acondicionamiento de señales electrocardiográficas abdominales maternas, aislamiento de la actividad cardíaca fetal y diagnóstico temprano de patologías cardíacas fetales mediante biomarcadores de variabilidad de frecuencia cardíaca (**fHRV**) y algoritmos de **Machine Learning**.

---

## 📌 Metodología y Flujo de Trabajo

El proyecto implementa un pipeline clínico-técnico de extremo a extremo dividido en cuatro etapas principales:

1. **Acondicionamiento y Filtrado Digital:**
   * Supresión de desplazamiento de línea base mediante filtro Butterworth Pasa-Altas ($0.5\text{ Hz}$, orden 3).
   * Atenuación de interferencia de red eléctrica mediante filtro Notch ($50/60\text{ Hz}$).
   * Supresión de ruido electromiográfico materno mediante filtro Pasa-Bajas ($100\text{ Hz}$, orden 4).
2. **Separación de Fuentes Ciegas (BSS) y Detección fQRS:**
   * Algoritmo **FastICA** para separar la señal fetal (fECG) de la señal materna predominante (mECG).
   * Selección de componente independiente óptima mediante criterio de máxima curtosis ($\gamma_2$).
   * Detección adaptativa de complejos ventriculares fetales ($fQRS$) mediante derivadas y umbrales móviles de Pan-Tompkins.
3. **Extracción de Biomarcadores Fisiológicos (fHRV):**
   * *Dominio del Tiempo:* $\text{BPM}_\text{mean}$, $\text{SDNN}$ (variabilidad global), $\text{RMSSD}$ (variabilidad a corto plazo), $\text{pNN20}$.
   * *Dominio de la Frecuencia:* Potencia $\text{LF}$ ($0.04 - 0.15\text{ Hz}$), $\text{HF}$ ($0.15 - 0.40\text{ Hz}$) y balance autonómico ($\text{LF/HF}$).
   * *Dinámica No Lineal:* Descriptores de Poincaré ($\text{SD1}$, $\text{SD2}$, ratio $\text{SD1/SD2}$) y Entropía Muestral ($\text{SampEn}$).
4. **Clasificación y Diagnóstico Probabilístico:**
   * Clasificador **Random Forest** balanceado con sobremuestreo sintético (**SMOTE**).
   * Calibración de umbral de decisión mediante el índice de Youden ($J$).

---

## 📂 Fuentes de Datos (PhysioNet)

* **[FECGSYNDB](https://physionet.org/content/fecgsyndb/1.0.0/):** Base de datos sintética biofísica que modela el entorno electrofisiológico materno-fetal bajo múltiples condiciones de relación señal/ruido ($SNR$), variabilidad autonómica y morfologías patológicas controladas.
* **[NIFEA DB](https://physionet.org/content/nifeadb/1.0.0/):** Base de datos clínica no invasiva de arritmias fetales con 26 registros reales (12 casos con patologías/arritmias diagnosticadas y 14 controles sanos).
* **[PhysioNet / Computing in Cardiology Challenge 2013 (set-a)](https://physionet.org/content/challenge-2013/1.0.0/):** Conjunto clínico estándar de 75 registros de ECG abdominal de 4 canales adquiridos en entornos hospitalarios reales, utilizados para validación cruzada y evaluación de detección de complejos fQRS.

---

## 🚀 Guía de Instalación y Uso

### 1. Clonar el repositorio

git clone https://github.com/**tu_usuario**/Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva.git
cd Detecci-n-de-Arritmias-Cardiacas-en-Fetos-de-manera-No-Invasiva


## 🛠️ Configuración del Entorno y Guía de Ejecución

Esta guía detalla el procedimiento paso a paso para configurar el entorno de trabajo, instalar las librerías necesarias y ejecutar cada módulo del pipeline.

---

### 1. Entorno Virtual de Python (Aislamiento del Proyecto)

#### ¿Por qué es necesario y recomendado?
Un entorno virtual crea un directorio aislado con su propia instalación de Python y gestor de paquetes (pip). 
* **Evita conflictos de versiones:** Garantiza que las versiones específicas de librerías como scikit-learn, xgboost o wfdb no interfieran con otros proyectos ni con las librerías globales del sistema operativo.
* **Reproducibilidad:** Asegura que cualquier colaborador que clone este repositorio ejecute el código bajo exactamente las mismas dependencias sin errores de compatibilidad.

#### Pasos para crearlo y activarlo:

1. **Creación del entorno (solo se hace la primera vez):**
   Abre una terminal en la raíz del proyecto y ejecuta:
   python -m venv venv

2. **Activación del entorno (debe hacerse cada vez que abras una nueva terminal):**
   * **Windows (PowerShell):**
     .\venv\Scripts\Activate.ps1
     (Si PowerShell bloquea la ejecución de scripts por políticas de seguridad, habilítalo ejecutando una vez: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser).
   * **Windows (CMD / Símbolo del sistema):**
     .\venv\Scripts\activate.bat
   * **Linux / macOS:**
     source venv/bin/activate

> **Verificación:** Sabrás que el entorno está activo si aparece el prefijo (venv) al inicio de la línea de comandos en tu terminal.

---

### 2. Instalación de Dependencias

#### ¿En qué momento y cómo hacerlo?
La instalación debe realizarse inmediatamente después de activar el entorno virtual por primera vez, y antes de ejecutar cualquier script de Python.

Ejecuta el siguiente comando en la terminal con el entorno (venv) activo:
pip install -r requirements.txt

Este comando leerá el archivo requirements.txt e instalará de forma automática todas las dependencias necesarias (numpy, scipy, pandas, wfdb, scikit-learn, xgboost, imbalanced-learn, joblib, streamlit, plotly, etc.) dentro de la carpeta aislada venv/.

---

### 3. Flujo de Ejecución del Pipeline

El sistema sigue una arquitectura modular. Para reproducir el flujo completo de ingeniería y ciencia de datos, ejecuta los módulos en el siguiente orden secuencial:

[1. Extracción de Features] ──> [2. Entrenamiento de Modelos] ──> [3. Inferencia de Prueba] ──> [4. Panel Web (Streamlit)]
   (build_dataset_full.py)             (model_trainer.py)             (pipeline_inference.py)               (app.py)

#### Paso 1: Extracción Masiva de Biomarcadores fHRV
Descarga y procesa en paralelo las señales de PhysioNet (NIFEA DB y FECGSYNDB), aplica filtrado digital, separación de fuentes ciegas (FastICA), detección fQRS y compila las matrices de características en un archivo CSV consolidado:
python src/build_dataset_full.py

* **Salida generada:** data/dataset_features_full.csv (o dataset_features.csv).

#### Paso 2: Entrenamiento, Validación Cruzada y Calibración
Entrena y compara múltiples arquitecturas (Random Forest, SVM RBF, Gradient Boosting) aplicando balanceo sintético de minorías (SMOTE), validación cruzada estratificada de 5 pliegues y calibración de umbral de decisión clínico (índice de Youden):
python src/model_trainer.py

* **Artefactos exportados en models/:**
  * detector_arritmias_fetal.pkl (modelo clasificador principal).
  * scaler_fhrv.pkl (escalador estadístico robusto).
  * feature_names.pkl (vector de variables seleccionadas).
  * decision_threshold.pkl (umbral óptimo de probabilidad).

#### Paso 3: Validación Unitaria de Inferencia Clínica
Prueba el pipeline de extremo a extremo procesando un registro clínico real (ARR_02) para validar la correcta carga de los artefactos y el cálculo de probabilidad diagnóstica en consola:
python src/pipeline_inference.py

#### Paso 4: Despliegue de la Interfaz Gráfica Interactiva
Inicia el panel web en Streamlit para cargar señales abdominales de 4 canales, visualizar el aislamiento de complejos fQRS, explorar dinámicas del tacograma RR / diagramas de Poincaré y obtener el reporte diagnóstico asistido por IA:
python -m streamlit run app.py

* **Acceso local:** Abre automáticamente tu navegador en http://localhost:8501.
