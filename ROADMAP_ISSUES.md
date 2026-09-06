# 📋 Plan de Trabajo y Gestión de Issues del Proyecto

Documento oficial de control de tareas y asignación de responsabilidades para el equipo de desarrollo (**Santiago Viasus** & **Juliana Bonilla**).

---

## 👥 Resumen del Reparto de Trabajo

| Issue | Nombre de la Tarea | Responsable Asignado | Estado |
| :---: | :--- | :---: | :---: |
| **#2** | Reentrenamiento y Calibración con Dataset Masivo | **Santiago** | ✅ **Completado (Cerrar)** |
| **#3** | Explicabilidad Clínica e Interpretabilidad con SHAP (XAI) | **Santiago** | ⏳ **Pendiente** |
| **#6** | Benchmark Cuantitativo de Detección fQRS (Se, PPV, F1) | **Santiago** | ⏳ **Pendiente** |
| **#4** | Generación y Descarga de Reportes Médicos en PDF / CSV | **Juliana** | ⏳ **Pendiente** |
| **#5** | Gráfico de Densidad Espectral PSD de Bandas Fetales | **Juliana** | 🟡 **75% Hecho (Falta PSD)** |
| **#11**| Módulo de Calidad de Señal Fetal (fSQI) y Detección de Ruido | **Juliana** | ⏳ **Pendiente (Nueva)** |

---

## 📌 Detalle de las Tareas por Responsable

### 🧑‍💻 Tareas de Santiago

#### 1. [ISSUE #3] Módulo de Explicabilidad Clínica (XAI con SHAP)
* **Objetivo:** Eliminar el efecto "caja negra" mostrando exactamente qué variables fisiológicas determinaron el diagnóstico.
* **Checklist:**
  - [ ] Instalar e importar `shap>=0.42.0` en el proyecto.
  - [ ] Crear `src/explainability.py` utilizando `shap.TreeExplainer`.
  - [ ] Generar gráfico de cascada (*Waterfall Plot*) que cuantifique el impacto de cada biomarcador (FCF, SDNN, Poincaré) en el paciente evaluado.
  - [ ] Integrar el panel explicativo dentro de `app.py`.

#### 2. [ISSUE #6] Benchmark de Detección fQRS sobre CinC 2013 (Set-A)
* **Objetivo:** Medir cuantitativamente la exactitud del detector de latidos frente a las anotaciones médicas oficiales.
* **Checklist:**
  - [ ] Crear el evaluador `src/evaluate_fqrs_detector.py`.
  - [ ] Comparar las marcas detectadas con los archivos `.fqrs` de los 75 pacientes con tolerancia temporal estándar de $\pm 50\text{ ms}$.
  - [ ] Reportar Sensibilidad ($Se$), Valor Predictivo Positivo ($PPV$) y puntuación $F_1$, con meta clínica $> 90\%$.

---

### 👩‍💻 Tareas de Juliana

#### 1. [ISSUE #4] Generación y Descarga de Reportes Médicos en PDF / CSV
* **Objetivo:** Permitir al médico descargar un informe clínico formal con un solo clic.
* **Checklist:**
  - [ ] Diseñar plantilla en PDF con membrete hospitalario (usando `reportlab` o `fpdf2`).
  - [ ] Incluir datos de la sesión, veredicto diagnóstico con color de semáforo, tabla de 13 biomarcadores e imagen del tacograma RR / Poincaré.
  - [ ] Integrar botones interactivos de descarga (`📥 Descargar Reporte PDF`) en `app.py`.

#### 2. [ISSUE #5] Gráfica de Densidad Espectral PSD en Streamlit
* **Objetivo:** Completar la visualización del balance simpático/parasimpático en la interfaz web.
* **Checklist:**
  - [ ] Calcular la Densidad Espectral de Potencia (PSD mediante método de Welch) sobre el tacograma RR.
  - [ ] Graficar en Plotly sombreando las bandas fisiológicas fetales: VLF ($< 0.04\text{ Hz}$), LF ($0.04 - 0.20\text{ Hz}$) y HF ($0.20 - 1.00\text{ Hz}$).

#### 3. [ISSUE #11] Módulo de Índice de Calidad de Señal Fetal (fSQI)
* **Objetivo:** Prevenir diagnósticos erróneos por ruido de movimiento o contracción materna antes de clasificar.
* **Checklist:**
  - [ ] Crear `src/signal_quality.py` calculando el índice **bSQI** (concordancia entre detectores) y **kSQI** (curtosis).
  - [ ] Definir zonas de calidad: Alta ($\ge 75\%$), Regular ($50\% - 75\%$) e Insuficiente ($< 50\%$).
  - [ ] Integrar una barra visual de calidad de señal en la barra superior de `app.py`.
