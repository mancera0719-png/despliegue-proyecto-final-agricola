import io
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predicción del Rendimiento Agrícola", layout="wide")

st.title("Predicción del Rendimiento Agrícola")
st.write("Esta aplicación predice el rendimiento de un cultivo en toneladas por hectárea (t/ha) "
         "usando el modelo Lasso entrenado en el proyecto final, ya sea ingresando los datos "
         "manualmente o subiendo un archivo Excel.")

COLUMNAS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_type"]


@st.cache_resource
def cargar_artefactos():
    try:
        modelo = joblib.load("modelo_rendimiento.joblib")
        with open("metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return modelo, metadata
    except Exception as e:
        st.error(f"Error al cargar los artefactos del modelo: {e}")
        return None, None


def limpiar(df):
    """Mismas reglas de limpieza del notebook: los valores imposibles pasan a vacío y el modelo los imputa."""
    df = df.copy()
    for col in COLUMNAS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[COLUMNAS]
    for col in COLUMNAS[:-1]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df.loc[(df["N"] < 0) | (df["N"] > 200), "N"] = np.nan
    df.loc[(df["ph"] < 3) | (df["ph"] > 10), "ph"] = np.nan
    df.loc[df["rainfall"] < 0, "rainfall"] = np.nan
    df["soil_type"] = df["soil_type"].astype(object).where(df["soil_type"].notna(), np.nan)
    return df


modelo, metadata = cargar_artefactos()

if modelo is not None and metadata is not None:
    rangos = metadata["rangos_entrenamiento"]
    intervalo = metadata["intervalo_residuos"]

    opcion = st.sidebar.selectbox("Selecciona el modo de entrada:",
                                  ["Predicción Individual", "Carga Masiva (Excel)"])

    if opcion == "Predicción Individual":
        st.header("Predicción Individual (Datos de Entrada)")

        col1, col2 = st.columns(2)
        with col1:
            n = st.number_input("Nitrógeno (N):", min_value=0.0, max_value=200.0,
                                value=float(rangos["N"]["mediana"]), step=1.0)
            p = st.number_input("Fósforo (P):", min_value=0.0, max_value=200.0,
                                value=float(rangos["P"]["mediana"]), step=1.0)
            k = st.number_input("Potasio (K):", min_value=0.0, max_value=250.0,
                                value=float(rangos["K"]["mediana"]), step=1.0)
            suelo = st.selectbox("Tipo de suelo:", metadata["tipos_de_suelo"])
        with col2:
            temp = st.number_input("Temperatura (°C):", min_value=0.0, max_value=50.0,
                                   value=round(float(rangos["temperature"]["mediana"]), 1), step=0.1)
            hum = st.number_input("Humedad (%):", min_value=0.0, max_value=100.0,
                                  value=round(float(rangos["humidity"]["mediana"]), 1), step=0.5)
            ph = st.number_input("pH del suelo:", min_value=3.0, max_value=10.0,
                                 value=round(float(rangos["ph"]["mediana"]), 2), step=0.1)
            lluvia = st.number_input("Lluvia (mm):", min_value=0.0, max_value=400.0,
                                     value=round(float(rangos["rainfall"]["mediana"]), 1), step=1.0)

        if st.button("Realizar Predicción"):
            try:
                df_input = pd.DataFrame([{"N": n, "P": p, "K": k, "temperature": temp, "humidity": hum,
                                          "ph": ph, "rainfall": lluvia, "soil_type": suelo}])
                prediccion = float(modelo.predict(limpiar(df_input))[0])
                inferior = max(0.0, prediccion + intervalo["q025"])
                superior = prediccion + intervalo["q975"]

                st.success(f"### El rendimiento estimado es: {prediccion:.2f} t/ha")
                st.info(f"Con un 95 % de confianza, el rendimiento estaría entre "
                        f"{inferior:.2f} y {superior:.2f} t/ha.")

                with st.expander("Ver datos enviados al modelo"):
                    st.dataframe(df_input)
            except Exception as e:
                st.error(f"Ocurrió un error durante la predicción: {e}")

    elif opcion == "Carga Masiva (Excel)":
        st.header("Predicción Masiva desde Archivo Excel")
        st.write("Sube un archivo de Excel (.xlsx) con las columnas `N`, `P`, `K`, `temperature`, "
                 "`humidity`, `ph`, `rainfall` y `soil_type`. Si una celda viene vacía, el modelo "
                 "la completa con la mediana de entrenamiento.")

        uploaded_file = st.file_uploader("Selecciona un archivo Excel", type=["xlsx"])

        if uploaded_file is not None:
            try:
                df_excel = pd.read_excel(uploaded_file)
                st.subheader("Datos Originales Subidos")
                st.dataframe(df_excel.head())

                faltantes = [c for c in ["N", "P", "K"] if c not in df_excel.columns]
                if faltantes:
                    st.error(f"El archivo no contiene las siguientes columnas requeridas: {faltantes}")
                else:
                    if st.button("Procesar y Predecir Archivo"):
                        predicciones = modelo.predict(limpiar(df_excel))

                        df_resultado = df_excel.copy()
                        df_resultado["Rendimiento_estimado_t_ha"] = np.round(predicciones, 3)
                        df_resultado["Minimo_95"] = np.round(np.maximum(0, predicciones + intervalo["q025"]), 3)
                        df_resultado["Maximo_95"] = np.round(predicciones + intervalo["q975"], 3)

                        st.success("¡Predicciones realizadas con éxito!")
                        st.subheader("Resultados Estimados")
                        st.dataframe(df_resultado)

                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine="openpyxl") as writer:
                            df_resultado.to_excel(writer, index=False, sheet_name="Predicciones")

                        st.download_button(
                            label="Descargar Excel con Predicciones",
                            data=output.getvalue(),
                            file_name="predicciones_rendimiento.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
            except Exception as e:
                st.error(f"Ocurrió un error al procesar el archivo Excel: {e}")
else:
    st.warning("Asegúrate de que los archivos 'modelo_rendimiento.joblib' y 'metadata.json' "
               "estén en la misma carpeta que app.py.")
