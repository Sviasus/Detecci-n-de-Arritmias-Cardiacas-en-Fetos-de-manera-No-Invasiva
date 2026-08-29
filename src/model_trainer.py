import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE

DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "dataset_features.csv")


def cargar_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el dataset en {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Excluir metadatos para la matriz de características
    columnas_excluidas = ["Registro", "Dataset", "Target"]
    features = [c for c in df.columns if c not in columnas_excluidas]
    
    X = df[features].copy()
    y = df["Target"].values
    
    # Imputación de posibles valores no numéricos por la mediana
    X = X.fillna(X.median()).values
    return X, y, features


def calibrar_umbral_f1(y_true, y_probas):
    precisiones, recalls, thresholds = precision_recall_curve(y_true, y_probas)
    # Cálculo seguro de F1 sin divisiones por cero
    f1_scores = 2 * (precisiones * recalls) / (precisiones + recalls + 1e-8)
    idx_optimo = np.argmax(f1_scores)
    
    if idx_optimo < len(thresholds):
        umbral_optimo = thresholds[idx_optimo]
    else:
        umbral_optimo = 0.50
    return float(np.clip(umbral_optimo, 0.40, 0.65))


def evaluar_modelos_cv(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    modelos = {
        "Random Forest (SMOTE)": RandomForestClassifier(
            n_estimators=150, max_depth=10, class_weight='balanced', random_state=42
        ),
        "Gradient Boosting (SMOTE)": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42
        ),
        "SVM (RBF + SMOTE)": SVC(
            kernel='rbf', C=2.0, probability=True, class_weight='balanced', random_state=42
        )
    }
    
    resultados = {}
    
    print("\n" + "=" * 65)
    print("      EVALUACIÓN 5-FOLD CV CON SMOTE Y CALIBRACIÓN F1-SCORE      ")
    print("=" * 65)
    
    for nombre, modelo in modelos.items():
        y_real_total = []
        y_proba_total = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 1. Escalado robusto
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # 2. Balanceo SMOTE en el pliegue de entrenamiento
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
            
            # 3. Ajuste
            modelo.fit(X_train_res, y_train_res)
            probas = modelo.predict_proba(X_val_scaled)[:, 1]
            
            y_real_total.extend(y_val)
            y_proba_total.extend(probas)
            
        y_real_total = np.array(y_real_total)
        y_proba_total = np.array(y_proba_total)
        
        # Calibración de umbral óptimo
        umbral = calibrar_umbral_f1(y_real_total, y_proba_total)
        y_pred_calibrado = (y_proba_total >= umbral).astype(int)
        
        acc = accuracy_score(y_real_total, y_pred_calibrado)
        roc = roc_auc_score(y_real_total, y_proba_total)
        f1_m = f1_score(y_real_total, y_pred_calibrado, average='macro')
        tn, fp, fn, tp = confusion_matrix(y_real_total, y_pred_calibrado).ravel()
        
        especificidad = tn / (tn + fp + 1e-8)
        sensibilidad = tp / (tp + fn + 1e-8)
        
        print(f"\n--- {nombre} ---")
        print(f"Umbral Calibrado : {umbral:.3f}")
        print(f"Accuracy         : {acc * 100:.2f}% | ROC-AUC: {roc * 100:.2f}%")
        print(f"F1-Macro         : {f1_m * 100:.2f}%")
        print(f"Especificidad (Controles Sanos): {tn}/{tn+fp} ({especificidad * 100:.1f}%)")
        print(f"Sensibilidad  (Arritmias)      : {tp}/{tp+fn} ({sensibilidad * 100:.1f}%)")
        
        resultados[nombre] = {
            "modelo_proto": modelo,
            "f1_macro": f1_m,
            "roc_auc": roc,
            "umbral": umbral
        }
        
    mejor_nombre = max(resultados, key=lambda k: resultados[k]["f1_macro"])
    return mejor_nombre, resultados[mejor_nombre]


def entrenar_y_exportar_final(X, y, feature_names, mejor_nombre, umbral_calibrado):
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    
    if "Random Forest" in mejor_nombre:
        modelo_definitivo = RandomForestClassifier(
            n_estimators=150, max_depth=10, class_weight='balanced', random_state=42
        )
    elif "Gradient" in mejor_nombre:
        modelo_definitivo = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42
        )
    else:
        modelo_definitivo = SVC(
            kernel='rbf', C=2.0, probability=True, class_weight='balanced', random_state=42
        )
        
    modelo_definitivo.fit(X_res, y_res)
    
    # Exportación de artefactos
    joblib.dump(modelo_definitivo, os.path.join(MODELS_DIR, "detector_arritmias_fetal.pkl"))
    joblib.dump(scaler_final, os.path.join(MODELS_DIR, "scaler_fhrv.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
    joblib.dump(umbral_calibrado, os.path.join(MODELS_DIR, "decision_threshold.pkl"))
    
    print("\n" + "=" * 65)
    print(f"Modelo Definitivo Seleccionado : {mejor_nombre}")
    print(f"Umbral de Decisión Calibrado  : {umbral_calibrado:.3f}")
    print(f"Artefactos exportados exitosamente en '{MODELS_DIR}/'")
    print("=" * 65)


if __name__ == "__main__":
    X, y, features = cargar_dataset(CSV_PATH)
    mejor_modelo, info = evaluar_modelos_cv(X, y)
    entrenar_y_exportar_final(X, y, features, mejor_modelo, info["umbral"])