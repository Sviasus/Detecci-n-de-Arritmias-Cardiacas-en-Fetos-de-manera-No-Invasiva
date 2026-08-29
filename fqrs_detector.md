# Reporte Técnico: Calibración y Optimización del Detector fQRS

---

## 1. Resumen Ejecutivo

El objetivo de esta fase fue diseñar, validar y calibrar el algoritmo de procesamiento digital de señales para la extracción no invasiva del electrocardiograma fetal (fECG) y la detección precisa de los complejos fQRS[cite: 1]. La calibración se ejecutó sobre los **75 registros** del *Set A* de la base de datos **CinC Challenge 2013** de PhysioNet, contrastando las detecciones automáticas contra las anotaciones manuales de referencia de expertos clínicos (`.fqrs`) bajo una ventana de tolerancia temporal de $\pm 50\text{ ms}$[cite: 1].

---

## 2. Comparativa Cuantitativa de Iteraciones

A lo largo del proceso se implementaron cuatro versiones del detector en `src/fqrs_detector.py`. Los resultados obtenidos sobre la totalidad del dataset (75 registros) fueron los siguientes:

| Iteración / Enfoque | Sensibilidad Promedio | Precisión Promedio | $F_1\text{-Score}$ Promedio | Registros $F_1 \ge 90\%$ | Estado / Decisión |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **V1. FastICA Puro + Welch** | $23.47\%$ | $29.84\%$ | $25.49\%$ | $2\text{ / }75$ | **Descartado:** Selección errónea de fuentes y fallo por inversión de signo. |
| **V2. FastICA Multicanal Adaptativo** | $49.96\%$ | $48.99\%$ | $49.33\%$ | $12\text{ / }75$ | **Descartado:** Persistencia de interferencia del complejo materno ($mQRS$). |
| **V3. Supresión $mQRS$ + Pan-Tompkins** | **$77.29\%$** | **$72.63\%$** | **$74.65\%$** | **$29\text{ / }75$** | **Seleccionado (Óptimo):** Mejor balance general de métricas. |
| **V4. Search-Back Forzado** | $76.66\%$ | $66.71\%$ | $71.08\%$ | $19\text{ / }75$ | **Descartado:** Inserción excesiva de falsos positivos en zonas con ruido. |

---

## 3. Detalle de Pruebas, Diagnóstico y Correcciones

### Iteración 1: FastICA Puro con Selección Espectral (Welch)
* **Hipótesis inicial:** Separar los 4 canales abdominales con FastICA y seleccionar automáticamente la componente con mayor densidad espectral en la banda fetal ($1.8\text{ – }3.0\text{ Hz}$)[cite: 1].
* **Resultado:** Funcionó de manera óptima en el registro aislado `a01` ($F_1 = 98.28\%$), pero colapsó al evaluar el conjunto completo ($F_1 = 25.49\%$).
* **Diagnóstico de fallos:**
  1. *Indeterminación de signo en ICA:* FastICA invierte aleatoriamente la polaridad de las fuentes; los picos negativos quedaban invisibles para el detector cuadrático.
  2. *Dominancia materna:* La madre aporta armónicos de gran energía en la banda de $2.0\text{ Hz}$, confundiendo al selector de densidad espectral.
  3. *Inestabilidad por valores nulos:* Presencia de muestras `NaN` en los registros de PhysioNet que detenían la descomposición matricial.

### Iteración 2: Evaluación Multicanal y Búsqueda de Regularidad Fisiológica
* **Correcciones aplicadas:** 
  * Limpieza e interpolación lineal previa de datos (`NaN` / `Inf`).
  * Evaluación de ambas polaridades ($+x(t)$ y $-x(t)$) en todas las fuentes independientes.
  * Reemplazo del selector previo por un evaluador post-detección basado en el ritmo fisiológico fetal ($110\text{ – }160\text{ bpm}$)[cite: 1].
* **Resultado:** El $F_1\text{-score}$ aumentó al $49.33\%$ ($+23.84\%$).
* **Limitación identificada:** FastICA por sí solo no elimina totalmente los complejos maternos gigantes ($mQRS$), generando falsos positivos recurrentes.

### Iteración 3: Cancelación Materna Explícita + Pan-Tompkins Adaptativo
* **Correcciones aplicadas:**
  * Implementación de una etapa previa de **supresión materna adaptativa**: localización de los picos $mQRS$ y aplicación de una máscara de atenuación suave con ventana de Hanning ($\pm 45\text{ ms}$).
  * Algoritmo de Pan-Tompkins ajustado con derivada de 5 puntos, integración móvil ($30\text{ ms}$) y umbralización por bloques de $2\text{ segundos}$ para absorber variaciones locales de amplitud.
* **Resultado:** El desempeño global alcanzó su punto óptimo: **$F_1 = 74.65\%$** ($+25.32\%$), logrando **29 registros por encima del 90%**.

### Iteración 4: Implementación de Rescate de Latidos (*Search-Back*)
* **Hipótesis:** Recuperar latidos en brechas $RR > 1.6 \times \text{mediana}$ bajando el umbral al 50%.
* **Resultado:** Incrementó la sensibilidad en casos aislados pero redujo la precisión global al $66.71\%$ y el $F_1$ al $71.08\%$ debido a la detección errónea de artefactos de movimiento.
* **Convergencia ICA:** Se corrigió el aviso `ConvergenceWarning` ajustando los parámetros a `max_iter=3000` y `tol=1e-3` en el algoritmo FastICA.

---

## 4. Decisiones Técnicas Finales

1. **Estructura del Pipeline de Extracción:** Mantener la arquitectura de la **Iteración 3**:
   $$\text{Entrada 4 Canales} \longrightarrow \text{Filtros (Pasa-banda + Notch)} \longrightarrow \text{Supresión } mQRS \longrightarrow \text{FastICA (3000 iter)} \longrightarrow \text{Pan-Tompkins Adaptativo}$$
2. **Generación de Tacogramas $RR$:** Para compensar los registros de baja calidad en etapas posteriores, los intervalos $RR$ pasan por un filtro fisiológico de consistencia ($250\text{ ms} \le RR \le 650\text{ ms}$) antes del cálculo de características.
3. **Paso a la Siguiente Fase:** Con el extractor fQRS estabilizado, se aprueba la transición a la **Etapa 2 (Extracción de Características fHRV)** y a la preparación del conjunto de datos para clasificación supervisada[cite: 1].