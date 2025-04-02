import streamlit as st 
import pandas as pd
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

# URL base del repositorio de GitHub (asegúrate de usar el raw link de los archivos)
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

st.write("¡BIENVENIDO! A través de este medio usted podrá estimar su consumo de energía durante el periodo de 2024.")

venta = st.number_input("Ingrese el valor promedio mensual ($) de su factura de energía:", min_value=0, step=1, format="%d")

area = st.radio("Seleccione el Área:", ["URBANO", "RURAL"])

if st.button("Predecir Consumo"):
    fila = {"VENTA": venta, "URBANO": 1 if area == "URBANO" else 0, "RURAL": 1 if area == "RURAL" else 0}
    data_pred = pd.DataFrame([fila])

    scaler_url = GITHUB_BASE_URL + "Estandarizacion.pkl"
    model_url = GITHUB_BASE_URL + "Trained_KNN_EDEQ.pkl"
    
    try:
        scaler = pickle.loads(pd.read_pickle(scaler_url))
        trained_KNN = pickle.loads(pd.read_pickle(model_url))
    except Exception as e:
        st.error(f"❌ Error al cargar los archivos: {e}")
        st.stop()

    new_data_normalized = scaler.transform(data_pred[['VENTA']])
    df_normalizado = pd.DataFrame(new_data_normalized, columns=['VENTA'])
    df_pred = pd.concat([data_pred.drop('VENTA', axis=1), df_normalizado], axis=1)

    trained_features = trained_KNN.feature_names_in_
    if not all(feature in df_pred.columns for feature in trained_features):
        st.error("❌ Error: Faltan columnas necesarias para la predicción.")
        st.stop()

    df_pred = df_pred[trained_features]
    resultado = np.round(trained_KNN.predict(df_pred)[0], 2)

    st.subheader("🔎 Resultado de la Predicción")
    if resultado > 209.25:
        st.error(f"⚠️ Consumo estimado: {resultado} kWh")
    else:
        st.success(f"✅ Consumo estimado: {resultado} kWh")

meses_orden = {"Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8, 
               "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12}

def plot_consumo_cliente(df, cliente_id, periodo):
    df_cliente = df[df['CLIENTE_ID'] == cliente_id]
    df_grouped = df.groupby(['AÑO', 'MES'])[['CONSUMO DE ENERGIA (kWh)', 'VALOR FACTURA ($)']].mean().reset_index()
    df_cliente_grouped = df_cliente.groupby(['AÑO', 'MES'])[['CONSUMO DE ENERGIA (kWh)', 'VALOR FACTURA ($)']].sum().reset_index()
    df_grouped['MES_ORDEN'] = df_grouped['MES'].map(meses_orden)
    df_cliente_grouped['MES_ORDEN'] = df_cliente_grouped['MES'].map(meses_orden)
    df_grouped = df_grouped.sort_values(['AÑO', 'MES_ORDEN'])
    df_cliente_grouped = df_cliente_grouped.sort_values(['AÑO', 'MES_ORDEN'])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_grouped['MES'], df_grouped['CONSUMO DE ENERGIA (kWh)'], marker='o', linestyle='-', label='Promedio General')
    ax.plot(df_cliente_grouped['MES'], df_cliente_grouped['CONSUMO DE ENERGIA (kWh)'], marker='s', linestyle='--', label=f'Cliente {cliente_id}')
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Consumo de Energía (kWh)')
    ax.set_title(f'Consumo de Energía del Cliente {cliente_id} vs Promedio')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

st.title("¿Cómo ha sido el comportamiento de su consumo de energía en el último año?🔎")
cliente_id = st.number_input("Ingrese su número de cuenta EDEQ (Codigo NIU):", min_value=0, step=1)
periodo = st.selectbox("Seleccione el período que desea analizar:", ['Mensual', 'Trimestral', 'Anual'])

df_urls = [GITHUB_BASE_URL + f"data_app{i}.csv" for i in range(1, 7)]
df_list = [pd.read_csv(url) for url in df_urls]
df = pd.concat(df_list, ignore_index=True)

if st.button("Generar Reporte"):
    resumen = plot_consumo_cliente(df, cliente_id, periodo)
    st.dataframe(resumen)
