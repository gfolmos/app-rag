# App que utiliza LangChain para hacer preguntas sobre los datos de un archivo pdf
# RAG (Retrieval-Augmented Generation)
# Autor: Gerardo Figueroa
# Fecha: 10/06/26
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Configuración de la página (ÚNICA Y AL INICIO)
st.set_page_config(layout="wide")

API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = API_KEY

# Agente LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=API_KEY
)

# Definición de la interfaz principal
col1, col2 = st.columns([1, 2])
with col1:
    # Asegúrate de que esta ruta sea correcta en tu repositorio de GitHub
    if os.path.exists("images/img_ia.png"):
        st.image("images/img_ia.png", width=150)
with col2:
    st.header("Analizador de Documentos RAG (Retrieval-Augmented Generation)")
    st.write("🚀 Utiliza un agente de IA que es realmente sorprendente!")

# Explicación del programa
with st.expander("Explicación del Programa"):
    st.write("""
        En las empresas se tienen cantidad de documentos, fichas de calidad, procedimientos, manuales, etc. utilizando la IA se ahorra mucho tiempo con esta herramienta. \n
        Como realizar la pregunta simple (ejemplos): \n
        'Muestra como limpiar el refrigerador', 'Muestra la función power cool', 'Muestra número de página de este contenido:...'. \n
        Nota Final: \n
        Al utilizar la IA por seguridad no se recomendaría analizar los documentos confidenciales.
        Para poder utilizar una herramienta IA como esta corriendo por internet, se puede correr localmente (on-premise)
        en una computadora moderna para que la información no salga por internet y esté segura la información. \n
        *La herramienta es experimental, tarda un poco en procesar*
        """)

# Selección de archivos PDF disponibles en la raíz
archivos_pdf = [f for f in os.listdir('.') if f.endswith('.pdf')]

if not archivos_pdf:
    st.error("No se encontraron archivos pdf en la carpeta actual.")
else:
    archivo_seleccionado = st.selectbox("Selecciona el archivo que deseas analizar:", archivos_pdf)
    
    # --- PROCESAMIENTO DEL PDF (Se ejecuta una sola vez al cargar o cambiar el archivo) ---
    with st.spinner("Procesando y preparando el documento..."):
        loader = PyPDFLoader(archivo_seleccionado)
        docs = loader.load()

        if not docs:
            st.error("El PDF está vacío o no se pudo leer.")
            st.stop() # Detiene la ejecución si el PDF falla
            
        # Fragmentar texto
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        if not splits:
            st.error("No se generaron fragmentos del documento. Revisa el PDF.")
            st.stop()

        try:
            # Inicializar embeddings y Base de datos de manera global para el archivo seleccionado
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        except Exception as e:
            st.error(f"Hubo un error al inicializar el procesador de embeddings: {e}")
            st.stop()

    # --- INTERFAZ DE USUARIO (Vista previa y Chat) ---
    with st.expander(f"Ver vista previa de {archivo_seleccionado}"):
        pdf_viewer(archivo_seleccionado)

    pregunta = st.chat_input("Haz una pregunta sobre los datos...")
    
    if pregunta:
        with st.spinner("El agente está pensando y analizando los datos..."):
            try:
                # Prompt del sistema
                system_prompt = (
                    "Eres un asistente experto en análisis de documentos.\n"
                    "Responde la pregunta del usuario utilizando únicamente el siguiente contexto proporcionado. "
                    "Si no sabes la respuesta, di que no la sabes.\n\n"
                    "Contexto:\n{context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                # Creación de cadenas de ejecución (Chains)
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                # Invocación de la IA
                resultado = rag_chain.invoke({"input": pregunta})
                
                # Mostrar resultados
                st.write(f"**Pregunta:** {pregunta}")
                st.subheader("Respuesta del Asistente:")
                st.success(resultado["answer"])

            except Exception as e:
                st.error(f"Hubo un error al procesar la consulta: {e}")
# Pie de página
#st.sidebar.info("Estado: Conectado a Groq | Memoria: Activa")
#******************* fin programa ppal ***************************

# ******************Definicion panel lateral*************
   
# *****************fin lateral ***********************

