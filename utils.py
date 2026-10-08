"""Lógica de predicción, separada de la interfaz para poder probarla sin Streamlit."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "modelo_rendimiento.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

COLUMNAS_NUM = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
COLUMNAS = COLUMNAS_NUM + ["soil_type"]

# Mismas reglas físicas que se usaron en la limpieza del notebook (sección 6.2):
# lo que queda fuera pasa a NaN y el pipeline lo imputa con la mediana de entrenamiento.
RANGOS_VALIDOS = {
    "N": (0, 200),
    "P": (0, None),
    "K": (0, None),
    "temperature": (-50, 60),
    "humidity": (0, 100),
    "ph": (3.0, 10.0),
    "rainfall": (0, None),
}


def cargar_modelo(ruta: Path = MODEL_PATH):
    return joblib.load(ruta)


def cargar_metadatos(ruta: Path = METADATA_PATH) -> dict:
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def rendimiento_medio(metadata: dict) -> float:
    """Rendimiento medio de entrenamiento (o la mediana si el metadata es antiguo)."""
    if "rendimiento_medio_train" in metadata:
        return float(metadata["rendimiento_medio_train"])
    perc = metadata["rendimiento_percentiles"]
    return float(perc["valor"][perc["percentil"].index(50)])


def preparar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Deja el DataFrame con las 8 columnas que espera el modelo.

    Convierte a número, aplica los rangos válidos y devuelve también cuántas
    celdas se pasaron a NaN por estar fuera de rango.
    """
    d = df.copy()
    for c in COLUMNAS:
        if c not in d.columns:
            d[c] = np.nan
    d = d[COLUMNAS]

    corregidos = 0
    for c in COLUMNAS_NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
        lo, hi = RANGOS_VALIDOS[c]
        fuera = pd.Series(False, index=d.index)
        if lo is not None:
            fuera |= d[c] < lo
        if hi is not None:
            fuera |= d[c] > hi
        corregidos += int(fuera.sum())
        d.loc[fuera, c] = np.nan

    d["soil_type"] = d["soil_type"].astype(object).where(d["soil_type"].notna(), np.nan)
    d["soil_type"] = d["soil_type"].map(lambda v: v.strip() if isinstance(v, str) else v)
    d.loc[d["soil_type"] == "", "soil_type"] = np.nan
    return d, corregidos


def predecir_con_intervalo(modelo, metadata: dict, entrada: dict) -> dict:
    """Predice una parcela y devuelve el intervalo empírico del 95 %.

    En `entrada`, los datos desconocidos van como NaN o None: el pipeline los imputa.
    """
    df, _ = preparar_datos(pd.DataFrame([{c: entrada.get(c, np.nan) for c in COLUMNAS}]))
    pred = float(modelo.predict(df)[0])
    iq = metadata["intervalo_residuos"]
    return {
        "prediccion": pred,
        "inferior": max(0.0, pred + float(iq["q025"])),
        "superior": pred + float(iq["q975"]),
        "referencia": rendimiento_medio(metadata),
    }


def predecir_lote(modelo, metadata: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Predice muchas parcelas a la vez. Devuelve el DataFrame original con tres columnas nuevas."""
    datos, corregidos = preparar_datos(df)
    pred = modelo.predict(datos)
    iq = metadata["intervalo_residuos"]
    salida = df.copy()
    salida["rendimiento_predicho_t_ha"] = np.round(pred, 3)
    salida["intervalo95_inferior"] = np.round(np.maximum(0.0, pred + float(iq["q025"])), 3)
    salida["intervalo95_superior"] = np.round(pred + float(iq["q975"]), 3)
    return salida, corregidos


def curva_sensibilidad(modelo, entrada: dict, variable: str, minimo: float, maximo: float,
                       puntos: int = 40) -> pd.DataFrame:
    """Rendimiento predicho al mover una sola variable y dejar las demás fijas."""
    valores = np.linspace(minimo, maximo, puntos)
    base = {c: entrada.get(c, np.nan) for c in COLUMNAS}
    df, _ = preparar_datos(pd.DataFrame([{**base, variable: v} for v in valores]))
    return pd.DataFrame({variable: valores, "rendimiento (t/ha)": modelo.predict(df)})
