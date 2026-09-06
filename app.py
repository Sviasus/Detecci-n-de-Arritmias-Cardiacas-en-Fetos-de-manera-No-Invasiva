import os
import sys
import tempfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wfdb

# Configuración de rutas internas
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from src.data_streamer import stream_nifeadb  # type: ignore
    from src.preprocessing import preprocesar_senal_multicanal  # type: ignore
    from src.fqrs_detector import extraer_fqrs_optimo  # type: ignore
    from src.feature_extraction import calcular_tacograma_rr, extraer_vector_caracteristicas_completo  # type: ignore
except (ImportError, ModuleNotFoundError):
    from data_streamer import stream_nifeadb  # type: ignore
    from preprocessing import preprocesar_senal_multicanal  # type: ignore
    from fqrs_detector import extraer_fqrs_optimo  # type: ignore
    from feature_extraction import calcular_tacograma_rr, extraer_vector_caracteristicas_completo  # type: ignore




st.set_page_config(
    page_title="CardioFetal AI | Detección No Invasiva de Arritmias",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
# Estilos CSS Estables (Sin fuentes externas que rompan React DOM)
st.markdown("""
    <style>
        html, body, [class*="css"] { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; 
        }
        .metric-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            margin-bottom: 10px;
        }
        .metric-title {
            color: #94a3b8;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        .metric-value {
            color: #f8fafc;
            font-size: 1.6rem;
            font-weight: 700;
        }
        .status-box-normal {
            background-color: rgba(16, 185, 129, 0.15);
            border: 1px solid #10b981;
            border-radius: 10px;
            padding: 16px;
            color: #34d399;
            font-size: 1.25rem;
            font-weight: 700;
            text-align: center;
        }
        .status-box-warning {
            background-color: rgba(245, 158, 11, 0.15);
            border: 1px solid #f59e0b;
            border-radius: 10px;
            padding: 16px;
            color: #fbbf24;
            font-size: 1.25rem;
            font-weight: 700;
            text-align: center;
        }
        .status-box-alert {
            background-color: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
            border-radius: 10px;
            padding: 16px;
            color: #f87171;
            font-size: 1.25rem;
            font-weight: 700;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


MODELS_DIR = os.path.join(CURRENT_DIR, "models")
CINC_LOCAL_DIR = os.path.join(CURRENT_DIR, "data", "cinc2013_real", "set-a")


@st.cache_resource
def cargar_artefactos():
    try:
        mod = joblib.load(os.path.join(MODELS_DIR, "detector_arritmias_fetal.pkl"))
        scl = joblib.load(os.path.join(MODELS_DIR, "scaler_fhrv.pkl"))
        fts = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        thr = joblib.load(os.path.join(MODELS_DIR, "decision_threshold.pkl"))
        return mod, scl, fts, thr
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None, None, 0.50


modelo, scaler, feature_names, default_threshold = cargar_artefactos()

# ----------------- BARRA LATERAL: ENTRADA DE DATOS -----------------
st.sidebar.title("🩺 Control de Análisis")
st.sidebar.markdown("---")

modo_entrada = st.sidebar.radio(
    "Fuente de datos:",
    [
        "Base de datos NIFEA (Stream)",
        "Base de datos CinC 2013 (Set-A y Set-B)",
        "Subir archivo propio (WFDB / CSV)"
    ]
)

sig_raw = None
fs = 1000.0
nombre_muestra = ""

if modo_entrada == "Base de datos NIFEA (Stream)":
    registros_nifea = (
        [f"NR_{i:02d}" for i in range(1, 15)] +
        [f"ARR_{i:02d}" for i in range(1, 13)]
    )
    registro_sel = st.sidebar.selectbox("Registro Fetal (NIFEA DB):", registros_nifea, index=0)
    nombre_muestra = registro_sel
    try:
        sig_raw, fs, header = stream_nifeadb(registro_sel)
    except Exception as e:
        st.sidebar.error(f"Error descargando registro: {e}")

elif modo_entrada == "Base de datos CinC 2013 (Set-A y Set-B)":
    cinc_set_tipo = st.sidebar.radio("Subconjunto CinC:", ["Set-A (a01 - a75)", "Set-B (b01 - b99)"])
    
    if "Set-A" in cinc_set_tipo:
        registros_cinc = [f"a{i:02d}" for i in range(1, 76)]
        registro_cinc_sel = st.sidebar.selectbox("Registro CinC 2013 (Set-A):", registros_cinc, index=10)
        dir_cinc_activo = os.path.join(CURRENT_DIR, "data", "cinc2013_real", "set-a")
    else:
        registros_cinc = [f"b{i:02d}" for i in range(1, 100)]
        registro_cinc_sel = st.sidebar.selectbox("Registro CinC 2013 (Set-B):", registros_cinc, index=0)
        dir_cinc_activo = os.path.join(CURRENT_DIR, "data", "cinc2013_real", "set-b")

    nombre_muestra = f"CinC 2013 - {registro_cinc_sel}"
    path_rec = os.path.join(dir_cinc_activo, registro_cinc_sel)
    
    if os.path.exists(f"{path_rec}.hea"):
        try:
            rec = wfdb.rdrecord(path_rec)
            sig_raw = np.array(rec.p_signal, dtype=np.float64, copy=True)
            fs = float(rec.fs)
            st.sidebar.success(f"Registro `{registro_cinc_sel}` cargado ({len(sig_raw)/fs:.0f}s, {fs:.0f} Hz).")
        except Exception as e:
            st.sidebar.error(f"Error leyendo registro CinC: {e}")
    else:
        st.sidebar.warning(f"No se encontró `{registro_cinc_sel}.hea`. Ejecuta la descarga previa de Set-B.")

else:
    st.sidebar.markdown("**Carga de Archivos Externos**")
    archivos_subidos = st.sidebar.file_uploader(
        "Selecciona los archivos (.dat y .hea juntos, o un archivo .csv):",
        accept_multiple_files=True,
        type=["dat", "hea", "csv"]
    )

    if archivos_subidos:
        csv_file = next((f for f in archivos_subidos if f.name.lower().endswith('.csv')), None)
        dat_file = next((f for f in archivos_subidos if f.name.lower().endswith('.dat')), None)
        hea_file = next((f for f in archivos_subidos if f.name.lower().endswith('.hea')), None)

        if csv_file:
            try:
                df_subido = pd.read_csv(csv_file)
                sig_raw = df_subido.select_dtypes(include=[np.number]).values
                nombre_muestra = csv_file.name
                fs = st.sidebar.number_input("Frecuencia de muestreo (Hz):", min_value=100.0, max_value=2000.0, value=1000.0)
            except Exception as e:
                st.sidebar.error(f"Error leyendo CSV: {e}")
        elif dat_file and hea_file:
            st.sidebar.info(f"Archivos listos: `{dat_file.name}` y `{hea_file.name}`")
            btn_procesar = st.sidebar.button("▶ Procesar Registro WFDB", type="primary")
            if btn_procesar or "cached_sig_raw" in st.session_state:
                if btn_procesar:
                    upload_tmp = os.path.join(tempfile.gettempdir(), "cardiofetal_wfdb")
                    os.makedirs(upload_tmp, exist_ok=True)
                    base_name = os.path.splitext(dat_file.name)[0]
                    p_dat = os.path.join(upload_tmp, dat_file.name)
                    p_hea = os.path.join(upload_tmp, hea_file.name)
                    with open(p_dat, "wb") as f_out:
                        f_out.write(dat_file.getbuffer())
                    with open(p_hea, "wb") as f_out:
                        f_out.write(hea_file.getbuffer())
                    try:
                        record = wfdb.rdrecord(os.path.join(upload_tmp, base_name))
                        st.session_state["cached_sig_raw"] = np.array(record.p_signal, dtype=np.float64, copy=True)
                        st.session_state["cached_fs"] = float(record.fs)
                        st.session_state["cached_name"] = base_name
                    except Exception as e:
                        st.sidebar.error(f"Error interpretando WFDB: {e}")

                if "cached_sig_raw" in st.session_state:
                    sig_raw = st.session_state["cached_sig_raw"]
                    fs = st.session_state["cached_fs"]
                    nombre_muestra = st.session_state["cached_name"]
        else:
            st.sidebar.warning("Para registros WFDB, sube tanto el archivo .dat como el .hea simultáneamente.")

tiempo_visualizar = st.sidebar.slider("Ventana de Señal a Visualizar (s):", min_value=3.0, max_value=20.0, value=8.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Calibración Diagnóstica")
umbral_clinico = st.sidebar.slider(
    "Umbral de Decisión Clínico",
    min_value=0.20,
    max_value=0.80,
    value=float(default_threshold) if default_threshold else 0.50,
    step=0.02,
    help="Define el punto de corte de probabilidad para alertar arritmias fetales."
)

# ----------------- PANEL PRINCIPAL -----------------
st.title("🩺 Sistema de Monitoreo y Detección de Arritmias Fetales")
st.markdown("Plataforma de procesamiento digital de señales bioeléctricas abdominales maternas (**ni-fECG**), aislamiento de complejos fetales (**FastICA**) y análisis multiparamétrico de **fHRV**.")

if sig_raw is not None:
    try:
        duracion_sec = len(sig_raw) / fs
        
        with st.spinner("Procesando señal biomédica..."):
            fqrs_peaks, fecg_ica, fuentes_todas, idx_fetal, sig_filt = extraer_fqrs_optimo(
                sig_raw, fs, return_intermediates=True
            )
            features_dict = extraer_vector_caracteristicas_completo(fqrs_peaks, fs)

        st.info(f"📋 **Muestra Activa:** `{nombre_muestra}` | **Duración Total:** `{duracion_sec:.1f} s` | **Frecuencia de Muestreo:** `{fs} Hz` | **Complejos fQRS Detectados:** `{len(fqrs_peaks)}`")

        if modelo is not None and features_dict is not None:
            df_feat = pd.DataFrame([features_dict])[feature_names]
            X_scaled = scaler.transform(df_feat.values)
            prob_patologia = float(modelo.predict_proba(X_scaled)[0, 1])
            es_patologico = prob_patologia >= umbral_clinico
            
            # Métricas Resumen
            col_diag, col_bpm, col_sdnn, col_rmssd = st.columns([1.4, 1, 1, 1])
            
            with col_diag:
                # Sistema de semáforo de certeza diagnóstica
                if prob_patologia < 0.35:
                    st.markdown('<div class="status-box-normal">✅ Ritmo Fetal Normal (Control)</div>', unsafe_allow_html=True)
                    st.caption(f"Probabilidad de patología: **{prob_patologia * 100:.1f}%** | Certeza Normal: **{(1-prob_patologia)*100:.1f}%**")
                elif prob_patologia < 0.65:
                    st.markdown('<div class="status-box-warning">🟡 Observación Clínica (Dudoso)</div>', unsafe_allow_html=True)
                    st.caption(f"Probabilidad de patología: **{prob_patologia * 100:.1f}%** | Requiere seguimiento continuo.")
                else:
                    st.markdown('<div class="status-box-alert">⚠️ ALERTA: Patología / Arritmia</div>', unsafe_allow_html=True)
                    st.caption(f"Probabilidad de patología: **{prob_patologia * 100:.1f}%** | Alerta por encima del umbral.")

            with col_bpm:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Frecuencia Cardíaca Fetal</div>
                        <div class="metric-value">{features_dict.get('BPM_mean', 0):.1f} <span style="font-size: 0.9rem;">bpm</span></div>
                    </div>
                """, unsafe_allow_html=True)

            with col_sdnn:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">SDNN (Variabilidad Total)</div>
                        <div class="metric-value">{features_dict.get('SDNN', 0):.1f} <span style="font-size: 0.9rem;">ms</span></div>
                    </div>
                """, unsafe_allow_html=True)

            with col_rmssd:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">RMSSD (Variación Corto Plazo)</div>
                        <div class="metric-value">{features_dict.get('RMSSD', 0):.1f} <span style="font-size: 0.9rem;">ms</span></div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("---")


            # Visualización de Señales
            st.subheader("📈 Señales Electrocardiográficas Abdominales y fECG Reconstruido")
            n_pts = int(min(tiempo_visualizar, duracion_sec) * fs)
            t_vec = np.linspace(0, n_pts / fs, n_pts)

            fig_signals = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                subplot_titles=(
                    "Canal Abdominal 1 (Señal Cruda / Filtrada)",
                    "Componente Fetal Aislado por FastICA (IC Fetal)",
                    "Detección de Complejos fQRS (Pan-Tompkins Fetal)"
                ),
                vertical_spacing=0.08
            )

            fig_signals.add_trace(go.Scatter(x=t_vec, y=sig_filt[:n_pts, 0], line=dict(color="#38bdf8", width=1.2), name="Abdominal Ch1"), row=1, col=1)
            fig_signals.add_trace(go.Scatter(x=t_vec, y=fecg_ica[:n_pts], line=dict(color="#a855f7", width=1.2), name="fECG ICA"), row=2, col=1)
            fig_signals.add_trace(go.Scatter(x=t_vec, y=fecg_ica[:n_pts], line=dict(color="#34d399", width=1.2), name="fQRS"), row=3, col=1)

            picos_plot = [p for p in fqrs_peaks if p < n_pts]
            if picos_plot:
                fig_signals.add_trace(
                    go.Scatter(
                        x=np.array(picos_plot) / fs,
                        y=fecg_ica[picos_plot],
                        mode="markers",
                        marker=dict(symbol="circle", size=8, color="#ef4444"),
                        name="Picos fQRS"
                    ),
                    row=3, col=1
                )

            fig_signals.update_layout(height=550, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
            st.plotly_chart(fig_signals, width='stretch')

            st.markdown("---")

            # Gráficas fHRV
            st.subheader("📊 Análisis de Variabilidad del Ritmo Cardíaco Fetal (fHRV)")
            rr_clean = calcular_tacograma_rr(fqrs_peaks, fs)

            col_taco, col_poincare, col_hist = st.columns(3)

            with col_taco:
                fig_taco = go.Figure()
                fig_taco.add_trace(go.Scatter(y=rr_clean, mode="lines+markers", marker=dict(size=3), line=dict(color="#38bdf8", width=1.2)))
                fig_taco.update_layout(title="Tacograma RR Fetal", xaxis_title="Latido", yaxis_title="RR (ms)", template="plotly_dark", height=340)
                st.plotly_chart(fig_taco, width='stretch')

            with col_poincare:
                if len(rr_clean) > 2:
                    fig_poinc = go.Figure()
                    fig_poinc.add_trace(go.Scatter(x=rr_clean[:-1], y=rr_clean[1:], mode="markers", marker=dict(color="#f472b6", size=4, opacity=0.7)))
                    fig_poinc.update_layout(title="Diagrama de Poincaré", xaxis_title="RR_n (ms)", yaxis_title="RR_{n+1} (ms)", template="plotly_dark", height=340)
                    st.plotly_chart(fig_poinc, width='stretch')

            with col_hist:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=rr_clean, nbinsx=30, marker_color="#fbbf24", opacity=0.8))
                fig_hist.update_layout(title="Distribución de Intervalos RR", xaxis_title="RR (ms)", yaxis_title="Frecuencia", template="plotly_dark", height=340)
                st.plotly_chart(fig_hist, width='stretch')

            with st.expander("🔬 Ver Tabla Detallada de Biomarcadores Fisiológicos"):
                df_display = pd.DataFrame([features_dict]).T.reset_index()
                df_display.columns = ["Biomarcador / Métrica", "Valor Calculado"]
                st.dataframe(df_display, width='stretch')


    except Exception as err:
        st.error(f"Error procesando la señal: {err}")
else:
    st.warning("Selecciona un registro de la base de datos o sube archivos locales desde la barra lateral izquierda para comenzar.")