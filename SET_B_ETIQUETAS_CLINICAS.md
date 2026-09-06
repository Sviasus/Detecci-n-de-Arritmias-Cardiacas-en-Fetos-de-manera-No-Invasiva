# SET_B_ETIQUETAS_CLINICAS: Validación Externa y Diagnóstico Comparativo

**Proyecto:** Detección No Invasiva de Arritmias Cardíacas Fetales (ni-fECG & fHRV)  
**Conjunto Evaluado:** PhysioNet Computing in Cardiology Challenge 2013 (Set-B: `b01` a `b100`)  
**Propósito:** Servir de referencia clínica documentada para comparar las predicciones del sistema con criterios médicos universales (ACOG / FIGO).  

---

## 1. Resumen Estadístico de Concordancia Externa

- **Total de Pacientes del Set B:** 100 registros reales de 4 canales abdominales (1 minuto c/u).
- **Registros con Señal Bioeléctrica Válida:** 95 / 100.
- **Concordancia Diagnóstica (IA vs Criterio Cardiológico):** **69.5%** (66 de 95 casos).
- **Umbral Clínico de Decisión:** 0.500 con Calibración Isotónica de Probabilidades.

---

## 2. Criterios Médicos Aplicados para el Etiquetado

1. **Ritmo Normal / Control:** Frecuencia cardíaca fetal basal entre $110	ext{ y }160	ext{ bpm}$, variabilidad normal ($SDNN \in [15, 65]\text{ ms}$) y dispersión elíptica regular en Poincaré.
2. **Bradicardia Fetal:** Frecuencia cardíaca fetal sostenida $< 110\text{ bpm}$.
3. **Taquicardia Fetal:** Frecuencia cardíaca fetal sostenida $> 160\text{ bpm}$.
4. **Irregularidad / Extrasístoles:** Dispersión perpendicular excesiva en Poincaré ($SD1 > 50\text{ ms}$ o $pNN50 > 45\%$) debida a latidos prematuros y pausas compensatorias.

---

## 3. Tabla Completa de los 100 Pacientes de Set B

| Registro | FCF Promedio (bpm) | SDNN (ms) | Diagnóstico Cardiológico Clínico | Probabilidad IA | Predicción IA | ¿Coincide? |
| :--- | :---: | :---: | :--- | :---: | :--- | :---: |
| **`b01`** | 95.4 | 26.7 | Bradicardia Fetal (95.4 bpm < 110) | **94.5%** | Patológico / Arritmia | ✅ Sí |
| **`b02`** | 158.1 | 97.0 | Irregularidad / Extrasístoles (SD1: 85.8 ms, pNN50: 65.1%) | **68.2%** | Patológico / Arritmia | ✅ Sí |
| **`b03`** | 151.3 | 85.6 | Irregularidad / Extrasístoles (SD1: 102.3 ms, pNN50: 78.2%) | **83.0%** | Patológico / Arritmia | ✅ Sí |
| **`b04`** | 88.5 | 33.4 | Bradicardia Fetal (88.5 bpm < 110) | **3.0%** | Normal / Control | ❌ No |
| **`b05`** | 142.4 | 121.9 | Irregularidad / Extrasístoles (SD1: 107.6 ms, pNN50: 63.8%) | **79.0%** | Patológico / Arritmia | ✅ Sí |
| **`b06`** | 163.0 | 53.5 | Taquicardia Fetal (163.0 bpm > 160) + Irregularidad / Extrasístoles (SD1: 56.4 ms, pNN50: 38.0%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b07`** | 149.3 | 125.0 | Irregularidad / Extrasístoles (SD1: 110.1 ms, pNN50: 69.4%) | **83.0%** | Patológico / Arritmia | ✅ Sí |
| **`b08`** | 77.3 | 52.6 | Bradicardia Fetal (77.3 bpm < 110) | **3.9%** | Normal / Control | ❌ No |
| **`b09`** | 141.9 | 117.1 | Irregularidad / Extrasístoles (SD1: 100.1 ms, pNN50: 63.5%) | **73.9%** | Patológico / Arritmia | ✅ Sí |
| **`b10`** | 166.8 | 74.7 | Taquicardia Fetal (166.8 bpm > 160) + Irregularidad / Extrasístoles (SD1: 69.4 ms, pNN50: 54.2%) | **98.0%** | Patológico / Arritmia | ✅ Sí |
| **`b11`** | 137.9 | 161.6 | Irregularidad / Extrasístoles (SD1: 124.9 ms, pNN50: 72.4%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b12`** | 149.1 | 118.8 | Irregularidad / Extrasístoles (SD1: 106.6 ms, pNN50: 66.3%) | **21.8%** | Normal / Control | ❌ No |
| **`b13`** | 123.2 | 148.4 | Irregularidad / Extrasístoles (SD1: 97.6 ms, pNN50: 32.1%) | **95.4%** | Patológico / Arritmia | ✅ Sí |
| **`b14`** | 93.0 | 61.7 | Bradicardia Fetal (93.0 bpm < 110) + Irregularidad / Extrasístoles (SD1: 62.5 ms, pNN50: 12.2%) | **79.8%** | Patológico / Arritmia | ✅ Sí |
| **`b15`** | 77.3 | 48.1 | Bradicardia Fetal (77.3 bpm < 110) | **54.8%** | Patológico / Arritmia | ✅ Sí |
| **`b16`** | 160.5 | 103.3 | Taquicardia Fetal (160.5 bpm > 160) + Irregularidad / Extrasístoles (SD1: 103.5 ms, pNN50: 70.8%) | **61.2%** | Patológico / Arritmia | ✅ Sí |
| **`b17`** | 165.0 | 103.1 | Taquicardia Fetal (165.0 bpm > 160) + Irregularidad / Extrasístoles (SD1: 105.3 ms, pNN50: 70.0%) | **19.5%** | Normal / Control | ❌ No |
| **`b18`** | 141.5 | 149.1 | Irregularidad / Extrasístoles (SD1: 112.9 ms, pNN50: 78.9%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b19`** | 137.0 | 154.8 | Irregularidad / Extrasístoles (SD1: 107.9 ms, pNN50: 61.7%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b20`** | 151.9 | 63.6 | Irregularidad / Extrasístoles (SD1: 63.7 ms, pNN50: 35.5%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b22`** | 154.7 | 88.3 | Irregularidad / Extrasístoles (SD1: 93.7 ms, pNN50: 68.1%) | **22.7%** | Normal / Control | ❌ No |
| **`b23`** | 157.3 | 88.7 | Irregularidad / Extrasístoles (SD1: 89.4 ms, pNN50: 71.1%) | **80.1%** | Patológico / Arritmia | ✅ Sí |
| **`b24`** | 146.3 | 118.7 | Irregularidad / Extrasístoles (SD1: 119.6 ms, pNN50: 68.4%) | **83.4%** | Patológico / Arritmia | ✅ Sí |
| **`b25`** | 151.6 | 93.1 | Irregularidad / Extrasístoles (SD1: 87.2 ms, pNN50: 70.2%) | **97.4%** | Patológico / Arritmia | ✅ Sí |
| **`b26`** | 95.8 | 44.6 | Bradicardia Fetal (95.8 bpm < 110) | **11.2%** | Normal / Control | ❌ No |
| **`b27`** | 157.6 | 107.3 | Irregularidad / Extrasístoles (SD1: 85.6 ms, pNN50: 62.4%) | **99.4%** | Patológico / Arritmia | ✅ Sí |
| **`b28`** | 88.0 | 33.9 | Bradicardia Fetal (88.0 bpm < 110) | **13.9%** | Normal / Control | ❌ No |
| **`b29`** | 148.0 | 127.1 | Irregularidad / Extrasístoles (SD1: 114.9 ms, pNN50: 62.5%) | **67.2%** | Patológico / Arritmia | ✅ Sí |
| **`b30`** | 82.5 | 27.6 | Bradicardia Fetal (82.5 bpm < 110) | **0.0%** | Normal / Control | ❌ No |
| **`b31`** | 153.7 | 115.3 | Irregularidad / Extrasístoles (SD1: 92.0 ms, pNN50: 65.2%) | **59.2%** | Patológico / Arritmia | ✅ Sí |
| **`b32`** | 158.3 | 101.3 | Irregularidad / Extrasístoles (SD1: 98.3 ms, pNN50: 74.3%) | **99.4%** | Patológico / Arritmia | ✅ Sí |
| **`b33`** | 127.7 | 180.7 | Irregularidad / Extrasístoles (SD1: 133.1 ms, pNN50: 69.9%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b34`** | 136.6 | 140.4 | Irregularidad / Extrasístoles (SD1: 121.2 ms, pNN50: 62.9%) | **73.2%** | Patológico / Arritmia | ✅ Sí |
| **`b35`** | 152.0 | 113.6 | Irregularidad / Extrasístoles (SD1: 97.2 ms, pNN50: 62.6%) | **77.7%** | Patológico / Arritmia | ✅ Sí |
| **`b36`** | 145.1 | 65.3 | Irregularidad / Extrasístoles (SD1: 77.1 ms, pNN50: 38.3%) | **79.1%** | Patológico / Arritmia | ✅ Sí |
| **`b37`** | 159.9 | 107.1 | Irregularidad / Extrasístoles (SD1: 92.2 ms, pNN50: 56.8%) | **52.1%** | Patológico / Arritmia | ✅ Sí |
| **`b38`** | 160.8 | 66.7 | Taquicardia Fetal (160.8 bpm > 160) + Irregularidad / Extrasístoles (SD1: 61.2 ms, pNN50: 47.9%) | **98.7%** | Patológico / Arritmia | ✅ Sí |
| **`b39`** | 132.1 | 159.2 | Irregularidad / Extrasístoles (SD1: 117.6 ms, pNN50: 71.4%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b40`** | 98.0 | 102.9 | Bradicardia Fetal (98.0 bpm < 110) | **10.1%** | Normal / Control | ❌ No |
| **`b41`** | 162.0 | 93.8 | Taquicardia Fetal (162.0 bpm > 160) + Irregularidad / Extrasístoles (SD1: 86.9 ms, pNN50: 73.8%) | **13.0%** | Normal / Control | ❌ No |
| **`b42`** | 154.3 | 120.2 | Irregularidad / Extrasístoles (SD1: 82.7 ms, pNN50: 57.6%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b43`** | 154.4 | 16.0 | Ritmo Sinusal Normal (154.4 bpm, SDNN: 16.0 ms) | **94.5%** | Patológico / Arritmia | ❌ No |
| **`b44`** | 106.4 | 120.7 | Bradicardia Fetal (106.4 bpm < 110) + Irregularidad / Extrasístoles (SD1: 93.5 ms, pNN50: 30.1%) | **48.1%** | Normal / Control | ❌ No |
| **`b45`** | 146.6 | 109.2 | Irregularidad / Extrasístoles (SD1: 94.7 ms, pNN50: 64.9%) | **25.1%** | Normal / Control | ❌ No |
| **`b46`** | 154.8 | 109.4 | Irregularidad / Extrasístoles (SD1: 95.0 ms, pNN50: 63.6%) | **98.6%** | Patológico / Arritmia | ✅ Sí |
| **`b47`** | 71.6 | 21.9 | Bradicardia Fetal (71.6 bpm < 110) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b48`** | 156.4 | 106.2 | Irregularidad / Extrasístoles (SD1: 100.3 ms, pNN50: 76.5%) | **68.8%** | Patológico / Arritmia | ✅ Sí |
| **`b49`** | 142.5 | 146.1 | Irregularidad / Extrasístoles (SD1: 114.1 ms, pNN50: 64.0%) | **47.7%** | Normal / Control | ❌ No |
| **`b50`** | 161.0 | 115.0 | Taquicardia Fetal (161.0 bpm > 160) + Irregularidad / Extrasístoles (SD1: 104.3 ms, pNN50: 66.3%) | **37.7%** | Normal / Control | ❌ No |
| **`b51`** | 131.9 | 94.9 | Irregularidad / Extrasístoles (SD1: 100.5 ms, pNN50: 53.9%) | **85.8%** | Patológico / Arritmia | ✅ Sí |
| **`b53`** | 142.0 | 64.0 | Irregularidad / Extrasístoles (SD1: 72.3 ms, pNN50: 33.3%) | **60.4%** | Patológico / Arritmia | ✅ Sí |
| **`b55`** | 122.9 | 194.4 | Irregularidad / Extrasístoles (SD1: 124.7 ms, pNN50: 61.0%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b56`** | 118.3 | 186.3 | Irregularidad / Extrasístoles (SD1: 121.7 ms, pNN50: 51.1%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b57`** | 165.2 | 96.7 | Taquicardia Fetal (165.2 bpm > 160) + Irregularidad / Extrasístoles (SD1: 94.9 ms, pNN50: 66.1%) | **97.4%** | Patológico / Arritmia | ✅ Sí |
| **`b58`** | 149.7 | 137.9 | Irregularidad / Extrasístoles (SD1: 98.5 ms, pNN50: 63.7%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b59`** | 87.0 | 29.3 | Bradicardia Fetal (87.0 bpm < 110) | **40.1%** | Normal / Control | ❌ No |
| **`b60`** | 105.7 | 181.3 | Bradicardia Fetal (105.7 bpm < 110) + Irregularidad / Extrasístoles (SD1: 111.3 ms, pNN50: 49.4%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b61`** | 146.4 | 11.3 | Ritmo Sinusal Normal (146.4 bpm, SDNN: 11.3 ms) | **100.0%** | Patológico / Arritmia | ❌ No |
| **`b62`** | 157.4 | 113.3 | Irregularidad / Extrasístoles (SD1: 96.0 ms, pNN50: 74.5%) | **96.6%** | Patológico / Arritmia | ✅ Sí |
| **`b63`** | 109.0 | 94.7 | Bradicardia Fetal (109.0 bpm < 110) + Irregularidad / Extrasístoles (SD1: 98.1 ms, pNN50: 25.7%) | **78.4%** | Patológico / Arritmia | ✅ Sí |
| **`b64`** | 145.0 | 123.0 | Irregularidad / Extrasístoles (SD1: 104.4 ms, pNN50: 53.9%) | **32.9%** | Normal / Control | ❌ No |
| **`b65`** | 144.3 | 108.5 | Irregularidad / Extrasístoles (SD1: 94.8 ms, pNN50: 62.4%) | **65.7%** | Patológico / Arritmia | ✅ Sí |
| **`b66`** | 83.5 | 31.5 | Bradicardia Fetal (83.5 bpm < 110) | **64.3%** | Patológico / Arritmia | ✅ Sí |
| **`b67`** | 158.8 | 79.4 | Irregularidad / Extrasístoles (SD1: 76.9 ms, pNN50: 62.4%) | **89.2%** | Patológico / Arritmia | ✅ Sí |
| **`b68`** | 147.0 | 97.2 | Irregularidad / Extrasístoles (SD1: 94.4 ms, pNN50: 58.3%) | **91.3%** | Patológico / Arritmia | ✅ Sí |
| **`b69`** | 146.5 | 154.9 | Irregularidad / Extrasístoles (SD1: 115.2 ms, pNN50: 67.0%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b70`** | 148.2 | 125.1 | Irregularidad / Extrasístoles (SD1: 94.9 ms, pNN50: 74.3%) | **64.3%** | Patológico / Arritmia | ✅ Sí |
| **`b71`** | 153.8 | 112.6 | Irregularidad / Extrasístoles (SD1: 89.6 ms, pNN50: 67.0%) | **67.7%** | Patológico / Arritmia | ✅ Sí |
| **`b72`** | 147.8 | 72.2 | Ritmo Sinusal Normal (147.8 bpm, SDNN: 72.2 ms) | **28.6%** | Normal / Control | ✅ Sí |
| **`b73`** | 155.4 | 121.9 | Irregularidad / Extrasístoles (SD1: 114.1 ms, pNN50: 69.2%) | **15.0%** | Normal / Control | ❌ No |
| **`b74`** | 141.9 | 142.3 | Irregularidad / Extrasístoles (SD1: 116.5 ms, pNN50: 67.8%) | **93.1%** | Patológico / Arritmia | ✅ Sí |
| **`b75`** | 165.8 | 100.2 | Taquicardia Fetal (165.8 bpm > 160) + Irregularidad / Extrasístoles (SD1: 96.4 ms, pNN50: 67.5%) | **62.1%** | Patológico / Arritmia | ✅ Sí |
| **`b76`** | 133.4 | 186.8 | Irregularidad / Extrasístoles (SD1: 116.3 ms, pNN50: 61.3%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b77`** | 95.5 | 62.6 | Bradicardia Fetal (95.5 bpm < 110) | **82.4%** | Patológico / Arritmia | ✅ Sí |
| **`b78`** | 134.6 | 151.1 | Irregularidad / Extrasístoles (SD1: 126.1 ms, pNN50: 75.7%) | **85.4%** | Patológico / Arritmia | ✅ Sí |
| **`b79`** | 94.5 | 49.8 | Bradicardia Fetal (94.5 bpm < 110) | **13.4%** | Normal / Control | ❌ No |
| **`b80`** | 129.7 | 157.6 | Irregularidad / Extrasístoles (SD1: 122.5 ms, pNN50: 74.1%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b81`** | 84.1 | 55.9 | Bradicardia Fetal (84.1 bpm < 110) | **1.2%** | Normal / Control | ❌ No |
| **`b82`** | 144.5 | 132.7 | Irregularidad / Extrasístoles (SD1: 135.0 ms, pNN50: 76.9%) | **20.4%** | Normal / Control | ❌ No |
| **`b83`** | 85.3 | 24.5 | Bradicardia Fetal (85.3 bpm < 110) | **3.0%** | Normal / Control | ❌ No |
| **`b84`** | 144.5 | 127.9 | Irregularidad / Extrasístoles (SD1: 114.0 ms, pNN50: 74.8%) | **52.9%** | Patológico / Arritmia | ✅ Sí |
| **`b86`** | 131.0 | 176.2 | Irregularidad / Extrasístoles (SD1: 124.7 ms, pNN50: 66.7%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b87`** | 92.3 | 45.6 | Bradicardia Fetal (92.3 bpm < 110) | **1.2%** | Normal / Control | ❌ No |
| **`b88`** | 158.8 | 112.8 | Irregularidad / Extrasístoles (SD1: 84.1 ms, pNN50: 52.3%) | **98.6%** | Patológico / Arritmia | ✅ Sí |
| **`b89`** | 145.5 | 63.1 | Irregularidad / Extrasístoles (SD1: 71.9 ms, pNN50: 41.0%) | **82.1%** | Patológico / Arritmia | ✅ Sí |
| **`b90`** | 107.4 | 112.4 | Bradicardia Fetal (107.4 bpm < 110) + Irregularidad / Extrasístoles (SD1: 101.3 ms, pNN50: 32.0%) | **79.3%** | Patológico / Arritmia | ✅ Sí |
| **`b91`** | 141.4 | 132.4 | Irregularidad / Extrasístoles (SD1: 100.4 ms, pNN50: 67.4%) | **82.5%** | Patológico / Arritmia | ✅ Sí |
| **`b92`** | 149.4 | 101.9 | Irregularidad / Extrasístoles (SD1: 109.6 ms, pNN50: 71.7%) | **25.7%** | Normal / Control | ❌ No |
| **`b93`** | 151.0 | 131.7 | Irregularidad / Extrasístoles (SD1: 116.5 ms, pNN50: 70.9%) | **36.8%** | Normal / Control | ❌ No |
| **`b94`** | 125.8 | 153.2 | Irregularidad / Extrasístoles (SD1: 116.3 ms, pNN50: 69.2%) | **100.0%** | Patológico / Arritmia | ✅ Sí |
| **`b95`** | 80.5 | 44.2 | Bradicardia Fetal (80.5 bpm < 110) | **22.7%** | Normal / Control | ❌ No |
| **`b96`** | 93.4 | 80.5 | Bradicardia Fetal (93.4 bpm < 110) + Irregularidad / Extrasístoles (SD1: 84.1 ms, pNN50: 32.3%) | **44.9%** | Normal / Control | ❌ No |
| **`b97`** | 162.3 | 122.4 | Taquicardia Fetal (162.3 bpm > 160) + Irregularidad / Extrasístoles (SD1: 110.5 ms, pNN50: 69.7%) | **81.7%** | Patológico / Arritmia | ✅ Sí |
| **`b98`** | 156.5 | 139.9 | Irregularidad / Extrasístoles (SD1: 119.7 ms, pNN50: 65.9%) | **38.2%** | Normal / Control | ❌ No |
| **`b99`** | 163.3 | 98.6 | Taquicardia Fetal (163.3 bpm > 160) + Irregularidad / Extrasístoles (SD1: 77.7 ms, pNN50: 57.7%) | **97.2%** | Patológico / Arritmia | ✅ Sí |

---

*Documento generado automáticamente para soporte clínico, defensa técnica y auditoría independiente.*  
