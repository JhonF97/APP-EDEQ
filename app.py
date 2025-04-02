import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
import io
import matplotlib.pyplot as plt

# 📌 **URL base de los archivos en GitHub (versión RAW)**
GITHUB_BASE_URL = "https://raw.githubusercontent.com/JhonF97/APP-EDEQ/main/"

# 📌 **Función para cargar archivos desde GitHub**
def load_github_file(url, is_pickle=False):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        if is_pickle:
            return pickle.load(io.BytesIO(response.content))
        else:
            return pd.read_csv(io.StringIO(response.text), sep=",", encoding="utf-8")  # Forzar separador de comas
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo desde GitHub: {url}\n{str(e)}")
        st.stop()

# 📌 **Cargar archivos desde GitHub**
scaler = load_github_file(GITHUB_BASE_URL + "Estandarizacion.pkl", is_pickle=True)
trained_KNN = load_github_file(GITHUB_BASE_URL + "Trained_KNN_EDEQ.pkl", is_pickle=True)
df = load_github_file(GITHUB_BASE_URL + "data_app.csv")

# 📌 **Verificar columnas del DataFrame**
st.write("Columnas disponibles en el dataset:", df.columns.tolist())

# 📌 **Función para graficar el consumo**
def plot_consumo_cliente(df, cliente_id, periodo):
    if "CLIENTE_ID" not in df.columns:
        st.error("❌ Error: La columna 'CLIENTE_ID' no se encuentra en el DataFrame.")
        return
    
    df_cliente = df[df['CLIENTE_ID'] == cliente_id]
    
    if df_cliente.empty:
        st.warning("⚠️ No se encontraron datos para el cliente ingresado.")
        return
    
    if periodo == 'Mensual':
        df_cliente_grouped = df_cliente.groupby(['AÑO', 'MES'])[['CONSUMO kWh', 'VENTA']].sum().reset_index()
    elif periodo == 'Trimestral':
        df_cliente_grouped = df_cliente.groupby(['AÑO', 'TRIMESTRE'])[['CONSUMO kWh', 'VENTA']].sum().reset_index()
    else:
        df_cliente_grouped = df_cliente.groupby(['AÑO'])[['CONSUMO kWh', 'VENTA']].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_cliente_grouped.index, df_cliente_grouped['CONSUMO kWh'], marker='s', linestyle='--', label=f'Cliente {cliente_id}')
    
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Consumo de Energía (kWh)')
    ax.set_title(f'Consumo de Energía del Cliente {cliente_id}')
    ax.legend()
    ax.grid()
    
    st.pyplot(fig)

st.title("🔎 ¿Cómo ha sido el comportamiento de su consumo de energía?")
cliente_id = st.number_input("Ingrese su número de cuenta EDEQ (Código NIU):", min_value=0, step=1)
periodo = st.selectbox("Seleccione el período que desea analizar:", ['Mensual', 'Trimestral', 'Anual'])

if st.button("Generar Reporte"):
    plot_consumo_cliente(df, cliente_id, periodo)
