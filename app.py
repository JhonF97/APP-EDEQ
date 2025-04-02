import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
import io
import matplotlib.pyplot as plt

# URL base del repositorio de GitHub (asegúrate de cambiar "usuario", "repositorio" y "rama")
GITHUB_BASE_URL = "https://raw.githubusercontent.com/JhonF97/APP-EDEQ/main/"

# URL de la imagen
image_url = "https://www.edeq.com.co/Portals/0/logo-edeq2.png?ver=H4Jt_3kPjOiTcmw9sxiSPA%3D%3D"

st.markdown(
    f"""
    <style>
    .img-container {{
        position: absolute;
        top: 10px;
        left: 5px;
    }}
    </style>
    <div class="img-container">
        <img src="{image_url}" width="250">
    </div>
    """,
    unsafe_allow_html=True
)

for _ in range(6):  
    st.write("")

st.title("⚡ ESTIMADOR DEL CONSUMO DE ENERGÍA EN EL QUINDÍO ⚡")
st.write("¡BIENVENIDO! Estime su consumo de energía para 2024.")

venta = st.number_input("Ingrese el valor promedio mensual ($) de su factura de energía:", min_value=0, step=1, format="%d")
area = st.radio("Seleccione el Área:", ["URBANO", "RURAL"])

if st.button("Predecir Consumo"):
    fila = {"VENTA": venta, "URBANO": 1 if area == "URBANO" else 0, "RURAL": 1 if area == "RURAL" else 0}
    data_pred = pd.DataFrame([fila])
    
    # Cargar el scaler
    scaler_url = GITHUB_BASE_URL + "Estandarizacion.pkl"
    response = requests.get(scaler_url)
    if response.status_code != 200:
        st.error("❌ Error: No se pudo descargar el archivo de normalización.")
        st.stop()
    scaler = pickle.load(io.BytesIO(response.content))
    
    new_data_normalized = scaler.transform(data_pred[['VENTA']])
    df_normalizado = pd.DataFrame(new_data_normalized, columns=['var1_normalizada'])
    df_pred = pd.concat([data_pred.drop('VENTA', axis=1), df_normalizado.rename(columns={'var1_normalizada': 'VENTA'})], axis=1)
    
    # Cargar el modelo
    model_url = GITHUB_BASE_URL + "Trained_KNN_EDEQ.pkl"
    response = requests.get(model_url)
    if response.status_code != 200:
        st.error("❌ Error: No se pudo descargar el modelo de predicción.")
        st.stop()
    trained_KNN = pickle.load(io.BytesIO(response.content))
    
    trained_features = trained_KNN.feature_names_in_
    if not all(feature in df_pred.columns for feature in trained_features):
        st.error("❌ Error: Faltan algunas columnas necesarias para la predicción.")
        st.stop()
    
    df_pred = df_pred[trained_features]
    resultado = np.round(trained_KNN.predict(df_pred)[0], 2)
    
    st.subheader("🔎 Resultado de la Predicción")
    if resultado > 209.25:
        st.error(f"⚠️ Consumo estimado: {resultado} kWh\n🔴 ¡CUIDADO! Su consumo está por ENCIMA del promedio.")
    else:
        st.success(f"✅ Consumo estimado: {resultado} kWh\n🟢 ¡Super bien! Su consumo está dentro del rango normal.")

st.title("¿Cómo ha sido el comportamiento de su consumo de energía en el último año?🔎")
cliente_id = st.number_input("Ingrese su número de cuenta EDEQ:", min_value=0, step=1)
periodo = st.selectbox("Seleccione el período a analizar:", ['Mensual', 'Trimestral', 'Anual'])

# Cargar los datos desde GitHub
dataframes = []
for i in range(1, 7):
    url = GITHUB_BASE_URL + f"data_app{i}.csv"
    response = requests.get(url)
    if response.status_code == 200:
        dataframes.append(pd.read_csv(io.StringIO(response.text)))
    else:
        st.error(f"❌ Error al cargar {url}")
        st.stop()

df = pd.concat(dataframes, ignore_index=True)

if st.button("Generar Reporte"):
    st.dataframe(df[df['CLIENTE_ID'] == cliente_id])
