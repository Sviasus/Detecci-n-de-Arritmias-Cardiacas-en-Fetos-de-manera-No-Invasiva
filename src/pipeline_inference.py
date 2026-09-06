import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from fqrs_detector import extraer_fqrs_optimo
from feature_extraction import calcular_tacograma_rr, extraer_vector_caracteristicas_completo

MODELS_DIR = PROJECT_ROOT / "models"


class DiagnosticoArritmiaFetalPipeline:
    def __init__(self, models_dir=None):
        base_models = Path(models_dir) if models_dir else MODELS_DIR
        modelo_path = base_models / "detector_arritmias_fetal.pkl"
        scaler_path = base_models / "scaler_fhrv.pkl"
        features_path = base_models / "feature_names.pkl"
        threshold_path = base_models / "decision_threshold.pkl"

        if not all(p.exists() for p in [modelo_path, scaler_path, features_path, threshold_path]):
            raise FileNotFoundError(
                f"Artefactos del modelo no encontrados en '{base_models}'. Ejecute src/model_trainer.py primero."
            )

        self.modelo = joblib.load(modelo_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        self.umbral_decision = joblib.load(threshold_path)

    def diagnosticar(self, sig_cruda_4ch, fs=1000):
        """
        Ejecuta el pipeline clínico completo sobre un registro de 4 canales.
        """
        # 1. Extracción de fQRS e intermediarios para visualización
        picos_fqrs, fecg_ica, fuentes, idx_fetal, sig_filt = extraer_fqrs_optimo(
            sig_cruda_4ch, fs, return_intermediates=True
        )
        if len(picos_fqrs) < 10:
            return {
                "error": "Señal insuficiente o calidad muy baja para identificar picos fetales.",
                "picos_fqrs": picos_fqrs,
                "fecg_ica": fecg_ica,
                "sig_filtrada": sig_filt,
                "tacograma_rr": np.array([])
            }


        # 2. Tacograma y fHRV
        tacograma_rr = calcular_tacograma_rr(picos_fqrs, fs)
        vector_completo = extraer_vector_caracteristicas_completo(picos_fqrs, fs)

        # 3. Vector de entrada para el modelo
        x_vec = np.array([[vector_completo.get(f, 0.0) for f in self.feature_names]])
        x_scaled = self.scaler.transform(x_vec)

        # 4. Inferencia probabilística
        prob_arritmia = float(self.modelo.predict_proba(x_scaled)[0, 1])
        es_arritmia = int(prob_arritmia >= self.umbral_decision)

        diagnostico_texto = "Patológico / Arritmia Fetal" if es_arritmia == 1 else "Ritmo Fetal Normal (Control)"

        return {
            "diagnostico": diagnostico_texto,
            "es_arritmia": es_arritmia,
            "probabilidad_arritmia": prob_arritmia * 100.0,
            "confianza": (prob_arritmia if es_arritmia == 1 else (1.0 - prob_arritmia)) * 100.0,
            "umbral_usado": self.umbral_decision,
            "picos_fqrs": picos_fqrs,
            "fecg_ica": fecg_ica,
            "fuentes": fuentes,
            "idx_fetal": idx_fetal,
            "sig_filtrada": sig_filt,
            "tacograma_rr": tacograma_rr,
            "metricas_fhrv": vector_completo
        }



if __name__ == "__main__":
    from data_streamer import stream_nifeadb

    pipeline = DiagnosticoArritmiaFetalPipeline()
    print("Probando inferencia clínica sobre paciente real ARR_02 (NIFEA DB)...")
    sig, fs, _ = stream_nifeadb("ARR_02")
    
    resultado = pipeline.diagnosticar(sig, fs)
    print("\n--- RESULTADO DE LA INFERENCIA ---")
    print(f"Diagnóstico   : {resultado['diagnostico']}")
    print(f"Probabilidad  : {resultado['probabilidad_arritmia']:.2f}% de Arritmia")
    print(f"Confianza     : {resultado['confianza']:.2f}%")
    print(f"BPM Promedio  : {resultado['metricas_fhrv']['BPM_mean']:.1f} bpm")
    print(f"SDNN          : {resultado['metricas_fhrv']['SDNN']:.1f} ms")