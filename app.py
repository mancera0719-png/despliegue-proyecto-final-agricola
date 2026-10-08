import warnings

import numpy as np
import pandas as pd
import streamlit as st

from utils import (COLUMNAS, METADATA_PATH, MODEL_PATH, cargar_metadatos, cargar_modelo,
                   curva_sensibilidad, predecir_con_intervalo, predecir_lote)

st.set_page_config(page_title="Rendimiento agrícola", page_icon="🌾", layout="wide")


@st.cache_resource
def get_modelo():
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        modelo = cargar_modelo()
    return modelo, [str(a.message) for a in avisos]


@st.cache_data
def get_metadatos():
    return cargar_metadatos()


if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.title("🌾 Predicción del rendimiento agrícola")
    st.error("Faltan los archivos del modelo. Revisa que la carpeta `models/` esté en el repositorio "
             "con `modelo_rendimiento.joblib` y `metadata.json`.")
    st.stop()

modelo, avisos_carga = get_modelo()
metadata = get_metadatos()
rangos = metadata["rangos_entrenamiento"]
nombre_modelo = metadata.get("modelo", "modelo")

ETIQUETAS = {
    "N": "Nitrógeno (N)", "P": "Fósforo (P)", "K": "Potasio (K)",
    "temperature": "Temperatura (°C)", "humidity": "Humedad relativa (%)",
    "ph": "pH del suelo", "rainfall": "Lluvia (mm)",
}
PASOS = {"N": 1.0, "P": 1.0, "K": 1.0, "temperature": 0.1, "humidity": 0.5, "ph": 0.1, "rainfall": 1.0}

# ----------------------------------------------------------------------- encabezado
st.title("🌾 Predicción del rendimiento agrícola")
st.markdown(
    "Calcula cuántas **toneladas por hectárea** podría dar una parcela según su suelo y su clima. "
    f"Detrás está el modelo **{nombre_modelo}**, el que mejor salió entre los siete que se compararon en el proyecto."
)
if avisos_carga:
    st.warning("El modelo se guardó con otra versión de scikit-learn. Instala la que indica `requirements.txt`.")

tab_una, tab_lote, tab_modelo = st.tabs(["🔮 Una parcela", "📂 Varias parcelas (CSV)", "📈 Sobre el modelo"])

# ----------------------------------------------------------------------- una parcela
with tab_una:
    col_in, col_out = st.columns([1.15, 1], gap="large")

    entrada = {}
    with col_in:
        st.subheader("Datos de la parcela")
        st.caption("Si no conoces algún dato, desmarca «Lo sé» y el modelo usará la mediana de entrenamiento.")

        def control(var):
            info = rangos[var]
            c1, c2 = st.columns([4, 1])
            with c2:
                st.write("")
                conocido = st.checkbox("Lo sé", value=True, key=f"usar_{var}")
            with c1:
                valor = st.slider(ETIQUETAS[var], float(info["min"]), float(info["max"]),
                                  float(info["mediana"]), PASOS[var], disabled=not conocido, key=f"v_{var}")
            return valor if conocido else np.nan

        with st.expander("🧪 Nutrientes del suelo", expanded=True):
            for v in ["N", "P", "K"]:
                entrada[v] = control(v)
        with st.expander("🌤️ Clima y pH", expanded=True):
            for v in ["temperature", "humidity", "ph", "rainfall"]:
                entrada[v] = control(v)
        with st.expander("🌱 Tipo de suelo"):
            c1, c2 = st.columns([4, 1])
            with c2:
                st.write("")
                suelo_ok = st.checkbox("Lo sé", value=True, key="usar_soil")
            with c1:
                suelo = st.selectbox("Tipo de suelo", metadata["tipos_de_suelo"],
                                     index=metadata["tipos_de_suelo"].index("Loamy")
                                     if "Loamy" in metadata["tipos_de_suelo"] else 0,
                                     disabled=not suelo_ok)
            entrada["soil_type"] = suelo if suelo_ok else np.nan

    with col_out:
        st.subheader("Resultado")
        r = predecir_con_intervalo(modelo, metadata, entrada)
        diferencia = r["prediccion"] - r["referencia"]
        st.metric("Rendimiento estimado", f"{r['prediccion']:.2f} t/ha",
                  delta=f"{diferencia:+.2f} t/ha frente al promedio")
        st.info(f"Lo más probable (95 %) es que quede **entre {r['inferior']:.2f} y {r['superior']:.2f} t/ha**.")

        faltan = [ETIQUETAS.get(k, "Tipo de suelo") for k, v in entrada.items()
                  if (isinstance(v, float) and np.isnan(v))]
        if faltan:
            st.caption("Datos rellenados con la mediana: " + ", ".join(faltan) + ".")

        st.markdown("##### ¿Qué pasa si cambio una variable?")
        var = st.selectbox("Variable a mover", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
                           format_func=lambda x: ETIQUETAS[x])
        curva = curva_sensibilidad(modelo, entrada, var, rangos[var]["min"], rangos[var]["max"])
        st.line_chart(curva, x=var, y="rendimiento (t/ha)", height=240, color="#3B7D3A")
        st.caption("Las demás variables se quedan como las pusiste. "
                   "Si la línea sale casi plana, esa variable apenas mueve el resultado.")

# ----------------------------------------------------------------------- lote CSV
with tab_lote:
    st.subheader("Predecir varias parcelas a la vez")
    st.markdown("Sube un CSV con estas columnas (las que falten o vengan vacías se imputan): "
                + ", ".join(f"`{c}`" for c in COLUMNAS) + ".")

    plantilla = pd.DataFrame([
        {"N": 80, "P": 60, "K": 100, "temperature": 24.0, "humidity": 60.0, "ph": 6.5, "rainfall": 170.0, "soil_type": "Loamy"},
        {"N": 20, "P": 30, "K": 40, "temperature": 30.0, "humidity": 45.0, "ph": 5.8, "rainfall": 90.0, "soil_type": "Sandy"},
        {"N": 120, "P": 130, "K": 180, "temperature": "", "humidity": 70.0, "ph": "", "rainfall": 220.0, "soil_type": ""},
    ])
    st.download_button("Descargar plantilla de ejemplo", plantilla.to_csv(index=False).encode("utf-8"),
                       "plantilla_parcelas.csv", "text/csv")

    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        try:
            df_in = pd.read_csv(archivo, sep=None, engine="python")
        except Exception as e:  # noqa: BLE001
            st.error(f"No pude leer el archivo: {e}")
            st.stop()

        reconocidas = [c for c in COLUMNAS if c in df_in.columns]
        if not reconocidas:
            st.error("El archivo no tiene ninguna de las columnas esperadas. Revisa la plantilla.")
        else:
            resultado, corregidos = predecir_lote(modelo, metadata, df_in)
            ausentes = [c for c in COLUMNAS if c not in df_in.columns]
            if ausentes:
                st.warning("Columnas que no venían y se imputaron: " + ", ".join(ausentes))
            if corregidos:
                st.warning(f"{corregidos} valores estaban fuera de rango físico y se trataron como faltantes.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Parcelas", f"{len(resultado):,}")
            c2.metric("Rendimiento medio", f"{resultado['rendimiento_predicho_t_ha'].mean():.2f} t/ha")
            c3.metric("Rango", f"{resultado['rendimiento_predicho_t_ha'].min():.2f} – "
                               f"{resultado['rendimiento_predicho_t_ha'].max():.2f}")
            st.dataframe(resultado, hide_index=True)
            st.download_button("Descargar resultados", resultado.to_csv(index=False).encode("utf-8"),
                               "predicciones_rendimiento.csv", "text/csv", type="primary")

# ----------------------------------------------------------------------- sobre el modelo
with tab_modelo:
    mp, base = metadata["metricas_prueba"], metadata["linea_base_prueba"]
    st.subheader("Qué tan bien predice")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{mp['R2']:.3f}")
    c2.metric("RMSE", f"{mp['RMSE']:.3f} t/ha")
    c3.metric("MAE", f"{mp['MAE']:.3f} t/ha")
    c4.metric("Mejora vs. la media", f"{100 * (1 - mp['RMSE'] / base['RMSE']):.1f} %")
    st.caption(f"Medido una sola vez sobre {metadata.get('n_test', 240)} parcelas de prueba que el modelo no vio al entrenar. "
               "Explica cerca de la mitad de la variación: sirve para orientar y comparar escenarios, "
               "no para dar una cifra exacta por parcela.")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("##### Variables que más pesan")
        imp = (pd.DataFrame([{"Variable": ETIQUETAS.get(k, "Tipo de suelo"), "Importancia": max(v["media"], 0)}
                             for k, v in metadata["importancia_permutacion"].items()])
               .sort_values("Importancia", ascending=False))
        st.bar_chart(imp, x="Variable", y="Importancia", horizontal=True, height=300, sort=False, color="#3B7D3A")
        st.caption("Cuánto empeora el error (t/ha) si se desordena cada variable. Casi todo sale de N, P y K.")
    with col_b:
        st.markdown("##### Los 7 modelos comparados")
        comp = pd.DataFrame(metadata["comparacion_modelos"]).rename(columns={
            "modelo": "Modelo", "cv_rmse": "RMSE (CV)", "cv_r2": "R² (CV)",
            "test_r2": "R² prueba", "test_rmse": "RMSE prueba", "test_mae": "MAE prueba"})
        st.dataframe(comp.round(3), hide_index=True)
        st.caption("Se eligió el de menor RMSE en validación cruzada de 5 pliegues.")

    st.markdown("##### Limitaciones")
    st.markdown(
        "- El dataset es didáctico (1.200 registros) y el 96 % es de Colombia.\n"
        "- Temperatura, humedad, pH, lluvia y tipo de suelo casi no aportaron en estos datos.\n"
        "- Faltan variables que seguramente explican el resto: cultivo, manejo, riego, fertilización."
    )
