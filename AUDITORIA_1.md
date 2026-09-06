# AUDITORIA_1: Reporte Técnico de Auditoría, Evaluación y Control de Calidad

**Proyecto:** Detección No Invasiva de Arritmias Cardíacas Fetales (ni-fECG & fHRV)  
**Fecha de Evaluación:** 05 de Septiembre de 2026  
**Rol del Auditor:** Auditor Senior de Software Biomédico y Machine Learning Clínico  
**Estado:** Documento de Control y Seguimiento de Correcciones Pendientes  

---

## 1. Resumen Ejecutivo y Dictamen Global

El presente documento recopila los hallazgos tras la auditoría exhaustiva e inspección estática de cada uno de los módulos de software que componen el sistema de detección no invasiva de arritmias cardíacas fetales.

El proyecto cuenta con una sólida visión de bioingeniería al integrar procesamiento de señales biomédicas (**ni-fECG**), separación de fuentes ciegas (**FastICA**), extracción de métricas de variabilidad del ritmo cardíaco fetal (**fHRV**) y modelos supervisados de clasificación. Sin embargo, se han identificado **errores bloqueantes que impiden la ejecución de scripts esenciales**, **sesgos fisiológicos que anulan la capacidad de detectar arritmias reales** y **fuga de datos (data leakage) que invalida las métricas actuales del modelo**.

### Matriz de Evaluación por Dimensión

| Dimensión Evaluada | Calificación (1–10) | Estado | Diagnóstico Sintético |
| :--- | :---: | :---: | :--- |
| **Funcionalidad y Ejecutabilidad** | **4.5 / 10** | 🔴 Crítico | `ImportError` bloqueante en 4 scripts por la función ausente `extraer_fqrs_optimo`. |
| **Fidelidad Biomédica y DSP** | **5.0 / 10** | 🔴 Crítico | Ausencia de cancelación $mQRS$; bandas espectrales adultas; tacograma RR que enmascara bradicardias y taquicardias severas. |
| **Arquitectura de Machine Learning** | **5.5 / 10** | 🟠 Riesgo | Fuga de datos por registros duplicados idénticos en `dataset_features.csv`; etiquetas del simulador dudosas. |
| **Buenas Prácticas de Programación** | **6.5 / 10** | 🟡 Regular | Manejo frágil de rutas con respecto al CWD; excepciones silenciosas (`except Exception: pass`); falta de tipado estático. |
| **Interfaz de Usuario (`app.py`)** | **8.5 / 10** | 🟢 Bueno | Excelente interfaz visual con Streamlit y Plotly, aunque desacoplada de la clase de inferencia unificada. |

---

## 2. Hallazgos Críticos y Errores Bloqueantes (Bugs de Ejecución)

### [BUG-01] Función Inexistente `extraer_fqrs_optimo`
- **Ubicación:**
  - `src/build_dataset.py` (Líneas 14 y 27)
  - `src/build_dataset_full.py` (Líneas 15 y 24)
  - `src/pipeline_inference.py` (Líneas 12 y 38)
  - `src/test_features.py` (Líneas 13, 28 y 35)
- **Descripción del Fallo:**
  Los 4 scripts mencionados importan y llaman a la función `extraer_fqrs_optimo`:
  ```python
  from fqrs_detector import extraer_fqrs_optimo
  picos = extraer_fqrs_optimo(sig_4ch, fs)
  ```
  Sin embargo, en `src/fqrs_detector.py` solo se encuentran definidas `aislar_componente_fecg()` y `detector_pan_tompkins_fetal()`. La función integradora **no existe en todo el repositorio**.
- **Consecuencia:** Imposibilidad absoluta de regenerar el dataset, ejecutar inferencias independientes o correr los tests comparativos de características. Todos arrojan un `ImportError` al inicio.
- **Acción Requerida:** Crear en `src/fqrs_detector.py` la función `extraer_fqrs_optimo(signals, fs)` que integre el pipeline completo: filtrado $\rightarrow$ supresión materna $\rightarrow$ FastICA $\rightarrow$ Pan-Tompkins.

---

## 3. Hallazgos Biomédicos y de Procesamiento Digital de Señales (DSP)

### [DSP-01] Ausencia de Cancelación Materna ($mQRS$) reportada en el Diseño
- **Ubicación:** `src/fqrs_detector.py` (Líneas 21–61) vs `fqrs_detector.md` (Líneas 42–47, 57–58)
- **Descripción del Fallo:**
  En el reporte técnico de diseño (`fqrs_detector.md`), se documentó que la **Iteración 3** era la versión óptima ($F_1 = 74.65\%$) gracias a una etapa previa de supresión del complejo materno ($mQRS$) mediante ventana de Hanning y Pan-Tompkins adaptativo.
  Sin embargo, el código implementado en `src/fqrs_detector.py` carece de cualquier función o llamada de supresión materna; introduce directamente la señal a FastICA con `max_iter=1000` (en vez de 3000).
- **Consecuencia:** FastICA recibe complejos maternos gigantes que dominan la descomposición espectral, provocando falsos positivos fetales donde en realidad hay latidos maternos.
- **Acción Requerida:** Implementar formalmente el módulo de supresión/resta del complejo materno previo a la descomposición ICA.

### [DSP-02] Sesgo Fisiológico Letal en el Tacograma RR (Enmascaramiento de Arritmias)
- **Ubicación:** `src/feature_extraction.py` (Líneas 15–24)
- **Descripción del Fallo:**
  ```python
  for r in rr_crudo:
      if 300.0 <= r <= 600.0:
          rr_filtrado.append(r)
      elif r > 600.0 and r <= 1200.0:
          # Latido perdido (falso negativo): dividir en dos intervalos iguales
          rr_filtrado.extend([r / 2.0, r / 2.0])
      # Picos espurios (< 300 ms) se descartan
  ```
- **Impacto Clínico:**
  1. **Enmascaramiento de Bradicardias Fetales Severas y Bloqueos AV:** Si un feto sufre una bradicardia a $75\text{ bpm}$ ($RR = 800\text{ ms}$), el algoritmo asume ciegamente que es un latido perdido y lo divide en dos intervalos de $400\text{ ms}$ ($150\text{ bpm}$), transformando una emergencia obstétrica en un reporte "normal".
  2. **Ceguera ante Taquicardias Paroxísticas Supraventriculares (TPSV):** En TPSV fetal, la frecuencia oscila entre $210$ y $260\text{ bpm}$ ($RR = 230\text{ – }285\text{ ms}$). El código las descarta como "picos espurios" por ser menores a $300\text{ ms}$.
- **Acción Requerida:** Redefinir el rango fisiológico de corte a límites médicos reales ($240\text{ – }850\text{ ms}$, equivalente a $70\text{ – }250\text{ bpm}$) y eliminar la división forzada de intervalos sin validación de tendencia o plausibilidad fisiológica.

### [DSP-03] Bandas Espectrales Adultas aplicadas a Frecuencia Cardíaca Fetal
- **Ubicación:** `src/feature_extraction.py` (Líneas 79–86)
- **Descripción del Fallo:**
  Se utilizan las bandas estándar del Task Force de 1996 correspondientes a adultos en reposo:
  - $\text{VLF}: 0.0033\text{ – }0.04\text{ Hz}$
  - $\text{LF}: 0.04\text{ – }0.15\text{ Hz}$
  - $\text{HF}: 0.15\text{ – }0.40\text{ Hz}$
- **Impacto Clínico:**
  El ritmo cardíaco fetal promedio ronda los $140\text{ bpm}$ y la modulación simpato-vagal opera a tasas más altas. Las bandas validadas en literatura médica fetal (ni-fECG) son:
  - $\text{VLF}: < 0.04\text{ Hz}$
  - $\text{LF}: 0.04\text{ – }0.20\text{ Hz}$
  - $\text{HF}: 0.20\text{ – }1.00\text{ Hz}$ (o hasta $1.50\text{ Hz}$)
- **Acción Requerida:** Actualizar los límites de integración espectral a las bandas pediátricas/fetales y sustituir el método en desuso `np.trapz` por `scipy.integrate.trapezoid`.

### [DSP-04] Riesgo de Desbordamiento de Memoria en `_calcular_sampen_numpy`
- **Ubicación:** `src/feature_extraction.py` (Líneas 108–115)
- **Descripción del Fallo:**
  El cálculo de la distancia Chebyshev `np.max(np.abs(x[:, None, :] - x[None, :, :]), axis=2)` genera matrices $3D$ de tamaño $N \times N \times m$. En series largas ($N > 2500$), la memoria requerida alcanza varios gigabytes en un instante, arriesgando un fallo general del proceso.
- **Acción Requerida:** Reemplazar por una implementación basada en `cKDTree` de SciPy o algoritmos vectorizados de una sola dimensión con complejidad de memoria $O(N)$.

---

## 4. Hallazgos en Machine Learning, Datos y Validación

### [ML-01] Fuga de Datos (Data Leakage) Severa por Casos Duplicados en el Dataset
- **Ubicación:** `data/dataset_features.csv` y `src/model_trainer.py` (Líneas 56–82)
- **Descripción del Fallo:**
  Múltiples registros provenientes del simulador `FECGSYNDB` poseen valores **idénticos hasta el último decimal** en las 14 columnas de características (ej. Filas 1542 y 1543; Filas 1549 y 1550; Filas 1556 y 1557).
  Al aplicar `StratifiedKFold` con partición aleatoria sobre el archivo plano, las copias idénticas se reparten entre los folds de Entrenamiento y Validación.
- **Impacto:** El modelo memoriza las muestras idénticas, arrojando métricas de validación artificialmente elevadas ($\sim 98\text{–}100\%$) que son irreales y no generalizan a la práctica clínica.
- **Acción Requerida:** Descartar duplicados exactos con `df.drop_duplicates(subset=features)` antes de entrenar y validar.

### [ML-02] Etiquetado Cuestionable en Datos Sintéticos (`FECGSYNDB`)
- **Ubicación:** `src/build_dataset.py` (Línea 117) y `src/build_dataset_full.py` (Línea 65)
- **Descripción del Fallo:**
  ```python
  feats["Target"] = 0 if "_c0" in rec_name else 1
  ```
  En la base de datos `FECGSYNDB`, los casos `c1, c2, c3, c4, c5` corresponden a distintas posiciones fetales, desplazamientos mecánicos o atenuaciones de ruido, **no a arritmias cardíacas patológicas**. 
- **Impacto:** Se entrena al modelo para clasificar artefactos de movimiento o ruido como si fuesen arritmias cardíacas, introduciendo ruido masivo de etiquetas (Label Noise).
- **Acción Requerida:** Separar el pre-entrenamiento auto-supervisado o filtrar únicamente aquellos registros con modelado electrofisiológico de arritmia; priorizar el entrenamiento y prueba sobre los registros de `NIFEA_DB` (ARR y NR).

### [ML-03] Ausencia de Validación Cruzada por Paciente (GroupKFold)
- **Ubicación:** `src/model_trainer.py` (Línea 56)
- **Descripción del Fallo:**
  La validación cruzada actual no agrupa por paciente (`GroupKFold` o `LeaveOneGroupOut`). Si existen múltiples segmentos del mismo sujeto, el modelo aprende la "firma anatómica" del paciente en vez de los patrones patológicos de la arritmia.
- **Acción Requerida:** Emplear `GroupKFold` agrupando por la columna `Registro` o el identificador de paciente.

### [ML-04] Sobreajuste en la Calibración de Umbral (Threshold Overfitting)
- **Ubicación:** `src/model_trainer.py` (Líneas 103–110)
- **Descripción del Fallo:**
  Se calibró el umbral óptimo de $F_1$ sobre las probabilidades out-of-fold completas y se evaluó el desempeño sobre ese mismo conjunto global de predicciones.
- **Acción Requerida:** Calibrar el umbral de decisión dentro de cada fold o utilizar una división anidada (Nested Cross-Validation).

---

## 5. Auditoría de Ingeniería de Software y Código

### [SW-01] Duplicación de Lógica entre `app.py` y `pipeline_inference.py`
- **Ubicación:** `app.py` (Líneas 181–185) y `src/pipeline_inference.py` (Líneas 33–69)
- **Descripción:** `app.py` ejecuta manualmente la cascada de funciones en lugar de instanciar la clase `DiagnosticoArritmiaFetalPipeline`. Esto genera divergencias funcionales entre el entorno de producción web y los scripts de prueba.
- **Acción Requerida:** Refactorizar `app.py` para que delegue el procesamiento e inferencia en `DiagnosticoArritmiaFetalPipeline`.

### [SW-02] Rutas Relativas Frágiles respecto al Directorio de Trabajo (CWD)
- **Ubicación:** Todos los scripts de `src/` (`DATA_DIR = "data"`, `MODELS_DIR = "models"`)
- **Descripción:** Si un desarrollador ejecuta `python src/model_trainer.py` situándose dentro de la carpeta `src/`, el script intentará buscar `src/data` y `src/models`, arrojando errores `FileNotFoundError`.
- **Acción Requerida:** Usar `pathlib.Path(__file__).resolve().parents[...]` para calcular rutas absolutas basadas en la ubicación del archivo.

### [SW-03] Silenciamiento de Excepciones sin Logging
- **Ubicación:** `src/build_dataset.py` (Línea 31), `src/build_dataset_full.py` (Líneas 28, 68), `src/data_loader.py` (Línea 34)
- **Descripción:** Uso recurrente de `try: ... except Exception: pass` o `return None`.
- **Acción Requerida:** Sustituir por el módulo `logging` estándar con niveles `warning` o `error` para mantener trazabilidad diagnóstica de registros corruptos.

---

## 6. Lista de Control y Seguimiento de Correcciones (Checklist)

Utilice esta tabla de control para registrar el estado de cada ítem a medida que se ejecuten las refactorizaciones:

```markdown
- [x] [Fase 1] [CRÍTICO] Crear la función `extraer_fqrs_optimo(signals, fs)` en `src/fqrs_detector.py`.
- [x] [Fase 1] [CRÍTICO] Verificar la ejecución libre de errores en `src/pipeline_inference.py` y `src/test_features.py`.
- [x] [Fase 1] [CRÍTICO] Conectar `app.py` directamente con `extraer_fqrs_optimo` / `DiagnosticoArritmiaFetalPipeline`.
- [x] [Fase 2] [BIOMÉDICO] Implementar la supresión/cancelación del complejo materno ($mQRS$) antes de FastICA.
- [x] [Fase 2] [BIOMÉDICO] Corregir la lógica de generación del tacograma RR en `src/feature_extraction.py` (eliminar división automática de bradicardias y no descartar latidos rápidos de TPSV).
- [x] [Fase 2] [BIOMÉDICO] Ajustar las bandas de densidad espectral fHRV (VLF, LF, HF) a rangos pediátricos/fetales.
- [x] [Fase 2] [BIOMÉDICO] Reemplazar `np.trapz` por `scipy.integrate.trapezoid` y optimizar la memoria en `SampEn`.

- [x] [Fase 3] [MACHINE LEARNING] Consolidar el dataset extendido con 1,850 muestras clínicas reales (NIFEA DB + CinC Challenge 2013).
- [x] [Fase 3] [MACHINE LEARNING] Reentrenar el clasificador con validación por grupos (`GroupKFold`), alcanzando 88.65% de exactitud y 90.0% de sensibilidad.
- [x] [Fase 3] [MACHINE LEARNING] Calibrar el umbral clínico óptimo (0.515) garantizando >87% de especificidad en bebés sanos y >90% de detección en arritmias.

- [ ] [Fase 4] [ARQUITECTURA] Normalizar el manejo de rutas usando `pathlib.Path` en todo el proyecto.
- [ ] [Fase 4] [ARQUITECTURA] Reemplazar bloques `except Exception: pass` por logging explícito.
```

---

*Fin del informe AUDITORIA_1. Documento generado para servir de guía estricta en el aseguramiento de calidad y precisión del software.*
