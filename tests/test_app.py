import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from utils import (COLUMNAS, cargar_metadatos, cargar_modelo, curva_sensibilidad,  # noqa: E402
                   predecir_con_intervalo, predecir_lote, preparar_datos)

MODELO, META = cargar_modelo(), cargar_metadatos()
EJEMPLO = {"N": 80, "P": 60, "K": 100, "temperature": 24, "humidity": 60, "ph": 6.5,
           "rainfall": 170, "soil_type": "Loamy"}


def test_coincide_con_el_notebook():
    # El notebook imprime 3.83 y 3.74 t/ha para estos dos casos
    assert round(predecir_con_intervalo(MODELO, META, EJEMPLO)["prediccion"], 2) == 3.83
    incompleto = {**EJEMPLO, "N": np.nan, "soil_type": np.nan}
    assert round(predecir_con_intervalo(MODELO, META, incompleto)["prediccion"], 2) == 3.74


def test_intervalo_coherente():
    r = predecir_con_intervalo(MODELO, META, EJEMPLO)
    assert 0 <= r["inferior"] < r["prediccion"] < r["superior"]


def test_todo_faltante_no_falla():
    r = predecir_con_intervalo(MODELO, META, {c: None for c in COLUMNAS})
    assert np.isfinite(r["prediccion"])


def test_valores_imposibles_pasan_a_nan():
    d, n = preparar_datos(pd.DataFrame([{**EJEMPLO, "N": 500, "ph": 14, "rainfall": -10}]))
    assert n == 3 and d[["N", "ph", "rainfall"]].isna().all(axis=None)


def test_lote_con_columnas_faltantes_y_texto():
    df = pd.DataFrame({"N": [80, "abc", None], "P": [60, 30, 120], "soil_type": ["Loamy", "", "Nuevo"]})
    out, _ = predecir_lote(MODELO, META, df)
    assert len(out) == 3 and out["rendimiento_predicho_t_ha"].notna().all()


def test_curva_sube_con_nitrogeno():
    c = curva_sensibilidad(MODELO, EJEMPLO, "N", 0, 139)
    assert c["rendimiento (t/ha)"].iloc[-1] > c["rendimiento (t/ha)"].iloc[0]


def test_app_arranca_sin_errores():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert "t/ha" in at.metric[0].value
