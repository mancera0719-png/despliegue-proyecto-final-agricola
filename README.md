# despliegue-proyecto-final-agricola

Predicción del rendimiento agrícola en toneladas por hectárea (t/ha) con el modelo Lasso del proyecto final de Machine Learning.

La app tiene dos modos, que se eligen en el menú de la izquierda:

- **Predicción Individual:** escribes N, P, K, temperatura, humedad, pH, lluvia y tipo de suelo, y te da el rendimiento estimado con su rango al 95 %.
- **Carga Masiva (Excel):** subes un `.xlsx` con esas mismas columnas y descargas el archivo con las predicciones. `ejemplo_parcelas.xlsx` sirve de modelo.

## Archivos

- `app.py`: la aplicación de Streamlit.
- `modelo_rendimiento.joblib`: el pipeline entrenado (imputación, escalado, one-hot y Lasso).
- `metadata.json`: rangos de las variables, tipos de suelo, métricas e intervalo de predicción.
- `requirements.txt`: librerías. scikit-learn va fijo en 1.6.1 porque con esa versión se guardó el modelo.
- `notebooks/`: el análisis completo y el entrenamiento.

## Resultados del modelo

R² = 0,488 · RMSE = 0,541 t/ha · MAE = 0,433 t/ha en el conjunto de prueba. Casi toda la predicción sale del nitrógeno, el fósforo y el potasio.
