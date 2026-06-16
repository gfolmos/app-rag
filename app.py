# App que utiliza LangChain para hacer preguntas sobre los datos de un archivo
# automatic analyzer
# Autor: Gerardo Figueroa
# Fecha: 08/06/26
import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. Configuración de la página (ÚNICA Y AL INICIO)
st.set_page_config(layout="wide")

API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = API_KEY

# Inicializar el modelo de Groq de forma global
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# --- Interfaz principal ---
col1, col2 = st.columns([1, 2])
with col1:
    if os.path.exists("images/img_ia.png"):
        st.image("images/img_ia.png", width=150)
with col2:
    st.header("Analizador Automático (Automatic Analyzer)")
    st.write("🚀 Utiliza un agente de IA que es realmente sorprendente!")

# Explicacion del programa
with st.expander("Explicación del Programa"):
    st.write("""
            Prácticamente esta herramienta es un analista eficaz, solo realiza la pregunta como si la realizaras al analista y ya no tendrás que esperar días para recibir los informes. \n
            Cómo realizar la pregunta simple (ejemplos): \n
            'Muestra la suma de ventas', 'Muestra las ventas por región', 'Muestra número de Total_Transacciones por género' (fijarse en el nombre de la columna). \n
            Nota: Si manda un error o no reconoce alguna palabra, reconstruye la pregunta. \n
            Nota Final: \n
            Al utilizar la IA no se recomendaría analizar los documentos de la empresa por propia política de la empresa o por seguridad.
            Para poder utilizar una herramienta como esta corriendo con un agente de IA por internet, se puede utilizar localmente (on-premise)
            en una computadora moderna para que la información no salga por internet y esté segura la información. \n
            """)

# 1ra: Seleccion archivos CSV
archivos_csv = [f for f in os.listdir('.') if f.endswith('.csv')]

if not archivos_csv:
    st.error("No se encontraron archivos CSV en la carpeta actual.")
else:
    archivo_seleccionado = st.selectbox("Selecciona el archivo que deseas analizar:", archivos_csv)
    df = pd.read_csv(archivo_seleccionado, encoding="utf-8-sig")

    # 2da PARTE: MOSTRAR CONTENIDO (Desplegable)
    with st.expander(f"Ver vista previa de {archivo_seleccionado}"):
        st.dataframe(df)

    # 3ra PARTE: INPUT DE PREGUNTA
    pregunta = st.chat_input("Haz una pregunta sobre los datos...")

    if pregunta:
        with st.spinner("El agente está pensando y analizando los datos..."):
            try:
                # PROMPT CRÍTICO: Obliga a Llama a no salirse del formato rígido de LangChain
                prefix_prompt = (
                    "You are working with a pandas dataframe in Python. The name of the dataframe is `df`.\n"
                    "Strictly follow the format: Thought -> Action -> Observation -> Final Answer.\n"
                    "Do not chat. Do not explain your actions. If you have the answer, output 'Final Answer:' followed immediately by your response.\n"
                    "Always respond in Spanish."
                )

                # Creamos el agente pasando el prompt personalizado en prefix
                agente = create_pandas_dataframe_agent(
                    llm, 
                    df, 
                    verbose=False, 
                    allow_dangerous_code=True,
                    handle_parsing_errors=True,
                    prefix=prefix_prompt
                )

                # Construimos la consulta forzando también formato limpio
                consulta_final = pregunta + " Si la respuesta involucra múltiples filas o datos estructurados, devuélvela en una tabla Markdown clara."
                
                # Ejecutar la consulta
                resultado = agente.invoke(consulta_final)

                # Mostrar la respuesta
                st.write(f"**Pregunta:** {pregunta}")
                st.subheader("Respuesta del Asistente:")
                st.success(resultado["output"])

            except Exception as e:
                st.error(f"Hubo un error al procesar la consulta: {e}")

# Pie de página
#st.sidebar.info("Esta app utiliza Groq Cloud para el procesamiento de lenguaje natural y Pandas para el análisis local.")
#st.sidebar.info("Estado: Conectado a Groq | Memoria: Activa")
#******************* fin programa ppal ***************************

# ******************Definicion panel lateral*************
   
# *****************fin lateral ***********************

