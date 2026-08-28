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
```bash
git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
cd tu-repositorio
