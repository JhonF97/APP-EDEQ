import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
import io
import matplotlib.pyplot as plt
import os

# 📌 **URL base de los archivos en GitHub (versión RAW)**
GITHUB_BASE_URL = "https://raw.githubusercontent.com/JhonF97/APP-EDEQ/main/" 

# 📌 **Función para cargar archivos desde GitHub**
def load_github_file(url, is_pickle=False):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza error si falla la descarga
        
        if is_pickle:
            return pickle.load(io.BytesIO(response.content))  # Cargar archivo pickle
        else:
            return pd.read_csv(io.StringIO(response.text))  # Cargar archivo CSV
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo desde GitHub: {url}\n{str(e)}")
        st.stop()

# 📌 **Cargar archivos desde GitHub**
scaler = load_github_file(GITHUB_BASE_URL + "Estandarizacion.pkl", is_pickle=True)
trained_KNN = load_github_file(GITHUB_BASE_URL + "Trained_KNN_EDEQ.pkl", is_pickle=True)

# 📌 **URL de la imagen**
image_url = "https://www.edeq.com.co/Portals/0/logo-edeq2.png?ver=H4Jt_3kPjOiTcmw9sxiSPA%3D%3D"

# 📌 **CSS para posicionar la imagen en la esquina superior izquierda**
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

# 🔹 **Agregar espacio con líneas en blanco**
for _ in range(6):  
    st.write("")

# 📌 **Título de la Aplicación**
st.title("⚡ ESTIMADOR DEL CONSUMO DE ENERGÍA EN EL QUINDÍO ⚡")

st.write("¡BIENVENIDO! A través de este medio usted podrá estimar su consumo de energía durante el periodo de 2024. También tendrá la posibilidad de consultar el informe del comportamiento de energia en su hogar y comparar las variaciones que ha tenido en los últimos periodos.")

# 📌 **Capturar entrada del usuario**
venta = st.number_input("Ingrese el valor promedio mensual ($) de su factura de energía:", min_value=0, step=1, format="%d")

# 📌 **Seleccionar el área**
area = st.radio("Seleccione el área de residencia:", ["URBANO", "RURAL"])

# 📌 **Botón para hacer la predicción**
if st.button("Predecir Consumo"):
    # 📌 **Crear el DataFrame con las variables dummy**
    fila = {
        "VENTA": venta,
        "URBANO": 1 if area == "URBANO" else 0,
        "RURAL": 1 if area == "RURAL" else 0
    }

    data_pred = pd.DataFrame([fila])

    # 📌 **Aplicar la normalización**
    variables_a_normalizar = data_pred[['VENTA']]
    new_data_normalized = scaler.transform(variables_a_normalizar)

    # 📌 **Agregar la columna normalizada al DataFrame**
    df_normalizado = pd.DataFrame(new_data_normalized, columns=['var1_normalizada'])
    df_pred = pd.concat([data_pred, df_normalizado], axis=1)

    # 📌 **Eliminar la variable original y renombrar la nueva**
    df_pred = df_pred.drop('VENTA', axis=1).rename(columns={'var1_normalizada': 'VENTA'})

    # 📌 **Filtrar las características correctas para la predicción**
    trained_features = trained_KNN.feature_names_in_

    if not all(feature in df_pred.columns for feature in trained_features):
        missing_features = set(trained_features) - set(df_pred.columns)
        st.error(f"❌ Error: Faltan las siguientes columnas: {missing_features}")
        st.stop()

    df_pred = df_pred[trained_features]  # Asegurar que solo se usen las columnas correctas

    # 📌 **Realizar la predicción**
    Y_fut = trained_KNN.predict(df_pred)
    resultado = np.round(Y_fut[0], 2)

    # 📌 **Mostrar el resultado con advertencias**
    st.subheader("🔎 Resultado de la Predicción")

    if resultado > 209.25:
        st.error(f"⚠️ Consumo estimado: {resultado} kWh\n🔴 ¡CUIDADO! Su consumo está por ENCIMA del promedio (209.25 kWh).")
    else:
        st.success(f"✅ Consumo estimado: {resultado} kWh\n🟢 ¡Super bien! Su consumo está dentro del rango normal.")
        

# Diccionario para ordenar los meses
meses_orden = {"Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8, 
               "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12}

def plot_consumo_cliente(df, cliente_id, periodo):
    df_cliente = df[df['CLIENTE_ID'] == cliente_id]
    
    if periodo == 'Mensual':
        df_grouped = df.groupby(['AÑO', 'MES'])[['CONSUMO DE ENERGIA (kWh)', 'VALOR FACTURA ($)']].mean().reset_index()
        df_cliente_grouped = df_cliente.groupby(['AÑO', 'MES'])[['CONSUMO DE ENERGIA (kWh)', 'VALOR FACTURA ($)']].sum().reset_index()
        df_grouped['MES_ORDEN'] = df_grouped['MES'].map(meses_orden)
        df_cliente_grouped['MES_ORDEN'] = df_cliente_grouped['MES'].map(meses_orden)
        df_grouped = df_grouped.sort_values(['AÑO', 'MES_ORDEN'])
        df_cliente_grouped = df_cliente_grouped.sort_values(['AÑO', 'MES_ORDEN'])
        df_grouped['Periodo'] = df_grouped['AÑO'].astype(str) + '-' + df_grouped['MES']
        df_cliente_grouped['Periodo'] = df_cliente_grouped['AÑO'].astype(str) + '-' + df_cliente_grouped['MES']
    
    elif periodo == 'Trimestral':
        df_grouped = df.groupby(['AÑO', 'TRIMESTRE'])[['CONSUMO DE ENERGIA (kWh)', 'VALOR FACTURA ($)']].mean().reset_index()
        df_cliente_grouped = df_cliente.groupby(['AÑO', 'TRIMESTRE'])[['CONSUMO DE ENERGIA (kWh)', 'VALOR FACTURA ($)']].sum().reset_index()
        df_grouped['Periodo'] = df_grouped['AÑO'].astype(str) + '-' + df_grouped['TRIMESTRE'].astype(str)
        df_cliente_grouped['Periodo'] = df_cliente_grouped['AÑO'].astype(str) + '-' + df_cliente_grouped['TRIMESTRE'].astype(str)
    
    elif periodo == 'Anual':
        df_grouped = df.groupby(['AÑO'])[['CONSUMO DE ENERGIA (kWh)', 'VALOR FACTURA ($)']].mean().reset_index()
        df_cliente_grouped = df_cliente.groupby(['AÑO'])[['CONSUMO DE ENERGIA (kWh)', 'VALOR FACTURA ($)']].sum().reset_index()
        df_grouped['Periodo'] = df_grouped['AÑO'].astype(str)
        df_cliente_grouped['Periodo'] = df_cliente_grouped['AÑO'].astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_grouped['Periodo'], df_grouped['CONSUMO DE ENERGIA (kWh)'], marker='o', linestyle='-', label='Promedio General')
    ax.plot(df_cliente_grouped['Periodo'], df_cliente_grouped['CONSUMO DE ENERGIA (kWh)'], marker='s', linestyle='--', label=f'Cliente {cliente_id}')
    
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Consumo de Energía (kWh)')
    ax.set_title(f'Consumo de Energía del Cliente {cliente_id} vs Promedio')
    ax.legend()
    ax.set_xticks(range(len(df_grouped)))
    ax.set_xticklabels(df_grouped['Periodo'], rotation=45)
    ax.grid()
    
    st.pyplot(fig)
    
    resumen = pd.merge(df_grouped[['Periodo', 'CONSUMO DE ENERGIA (kWh)']], 
                        df_cliente_grouped[['Periodo', 'CONSUMO DE ENERGIA (kWh)', 'VALOR FACTURA ($)']], 
                        on='Periodo', suffixes=('_Promedio', '_Cliente'))

    return resumen.rename(columns={
        'CONSUMO DE ENERGIA (kWh)_Promedio': 'Consumo Promedio (kWh)',
        'CONSUMO DE ENERGIA (kWh)_Cliente': 'Consumo Cliente (kWh)',
        'VALOR FACTURA ($)': 'Valor Factura Cliente ($)'
    })


st.title("¿ Cómo ha sido el comportamiento de su consumo de energía en el último año?🔎")
cliente_id = st.number_input("Ingrese su número de cuenta EDEQ (Codigo NIU) el cual puedes ubicar en la parte superior derecha de la factura:", min_value=0, step=1)
periodo = st.selectbox("Seleccione el período que desea analizar:", ['Mensual', 'Trimestral', 'Anual'])

#cargar dataframe

archivo_csv1 = os.path.join(os.getcwd(), 'data_app1.csv')
archivo_csv2 = os.path.join(os.getcwd(), 'data_app2.csv')
archivo_csv3 = os.path.join(os.getcwd(), 'data_app3.csv')
archivo_csv4 = os.path.join(os.getcwd(), 'data_app4.csv')
archivo_csv5 = os.path.join(os.getcwd(), 'data_app5.csv')
archivo_csv6 = os.path.join(os.getcwd(), 'data_app6.csv')

# Leer el archivo CSV

df1 = pd.read_csv(archivo_csv1)
df2 = pd.read_csv(archivo_csv2)
df3 = pd.read_csv(archivo_csv3)
df4 = pd.read_csv(archivo_csv4)
df5 = pd.read_csv(archivo_csv5)
df6 = pd.read_csv(archivo_csv6)

df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

if st.button("Generar Reporte"):
    resumen = plot_consumo_cliente(df, cliente_id, periodo)
    st.dataframe(resumen)
