import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen

# 📌 **Definir la URL base del repositorio en GitHub**
GITHUB_BASE_URL = "https://raw.githubusercontent.com/JhonF97/APP-EDEQ/main/"

# URL de la imagen
image_url = "https://www.edeq.com.co/Portals/0/logo-edeq2.png?ver=H4Jt_3kPjOiTcmw9sxiSPA%3D%3D"

# CSS para posicionar la imagen en la esquina superior izquierda
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

st.title("⚡ ESTIMADOR DEL CONSUMO DE ENERGÍA EN EL QUINDÍO ⚡")

st.write("¡BIENVENIDO! A través de este medio usted podrá estimar su consumo de energía durante el periodo de 2024...")

# 📌 **Capturar la entrada del usuario**
venta = st.number_input("Ingrese el valor promedio mensual ($) de su factura de energía:", min_value=0, step=1, format="%d")
area = st.radio("Seleccione el Área:", ["URBANO", "RURAL"])

# 📌 **Botón para hacer la predicción**
if st.button("Predecir Consumo"):
    fila = {
        "VENTA": venta,
        "URBANO": 1 if area == "URBANO" else 0,
        "RURAL": 1 if area == "RURAL" else 0
    }
    data_pred = pd.DataFrame([fila])

    # 📌 **Cargar el scaler desde GitHub**
    scaler_url = GITHUB_BASE_URL + "Estandarizacion.pkl"
    try:
        with urlopen(scaler_url) as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error al cargar el scaler: {e}")
        st.stop()

    variables_a_normalizar = data_pred[['VENTA']]
    new_data_normalized = scaler.transform(variables_a_normalizar)
    df_pred = pd.concat([data_pred, pd.DataFrame(new_data_normalized, columns=['VENTA'])], axis=1).drop('VENTA', axis=1)

    # 📌 **Cargar el modelo desde GitHub**
    model_url = GITHUB_BASE_URL + "Trained_KNN_EDEQ.pkl"
    try:
        with urlopen(model_url) as f:
            trained_KNN = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()

    trained_features = trained_KNN.feature_names_in_
    if not all(feature in df_pred.columns for feature in trained_features):
        missing_features = set(trained_features) - set(df_pred.columns)
        st.error(f"❌ Error: Faltan las siguientes columnas: {missing_features}")
        st.stop()
    
    df_pred = df_pred[trained_features]
    resultado = np.round(trained_KNN.predict(df_pred)[0], 2)

    st.subheader("🔎 Resultado de la Predicción")
    if resultado > 209.25:
        st.error(f"⚠️ Consumo estimado: {resultado} kWh\n🔴 ¡CUIDADO! Su consumo está por ENCIMA del promedio (209.25 kWh).")
    else:
        st.success(f"✅ Consumo estimado: {resultado} kWh\n🟢 ¡Super bien! Su consumo está dentro del rango normal.")

# 📌 **Cargar los archivos de datos desde GitHub**
archivos_csv = ["data_app1.csv", "data_app2.csv", "data_app3.csv", "data_app4.csv", "data_app5.csv", "data_app6.csv"]
dataframes = []
for archivo in archivos_csv:
    url = GITHUB_BASE_URL + archivo
    try:
        df = pd.read_csv(url)
        dataframes.append(df)
    except Exception as e:
        st.error(f"❌ Error al cargar {archivo}: {e}")

if dataframes:
    df = pd.concat(dataframes, ignore_index=True)

    st.title("🔎 ¿Cómo ha sido el comportamiento de su consumo de energía en el último año?")
    cliente_id = st.number_input("Ingrese su número de cuenta EDEQ (Código NIU):", min_value=0, step=1)
    periodo = st.selectbox("Seleccione el período que desea analizar:", ['Mensual', 'Trimestral', 'Anual'])

    if st.button("Generar Reporte"):
        def plot_consumo_cliente(df, cliente_id, periodo):
            df_cliente = df[df['CLIENTE_ID'] == cliente_id]
            df_grouped = df.groupby(['AÑO', 'MES'])[['CONSUMO DE ENERGIA (kWh)', 'VALOR FACTURA ($)']].mean().reset_index()
            df_cliente_grouped = df_cliente.groupby(['AÑO', 'MES'])[['CONSUMO DE ENERGIA (kWh)', 'VALOR FACTURA ($)']].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_grouped['MES'], df_grouped['CONSUMO DE ENERGIA (kWh)'], marker='o', linestyle='-', label='Promedio General')
            ax.plot(df_cliente_grouped['MES'], df_cliente_grouped['CONSUMO DE ENERGIA (kWh)'], marker='s', linestyle='--', label=f'Cliente {cliente_id}')
            ax.set_xlabel('Mes')
            ax.set_ylabel('Consumo de Energía (kWh)')
            ax.set_title(f'Consumo de Energía del Cliente {cliente_id} vs Promedio')
            ax.legend()
            ax.grid()
            st.pyplot(fig)
        plot_consumo_cliente(df, cliente_id, periodo)
else:
    st.error("❌ No se pudieron cargar los datos del consumo.")
