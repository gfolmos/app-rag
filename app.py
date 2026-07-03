# App que utiliza LangChain para hacer preguntas sobre los datos de un archivo pdf
# RAG (Retrieval-Augmented Generation)
# Autor: Gerardo Figueroa
# Fecha: 10/06/26
# rag optimizado
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import os
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(layout="wide", page_title="Analizador RAG", page_icon="🤖")

# 2. INICIALIZACIÓN DE ENTORNO Y LLM
@st.cache_resource
def inicializar_llm():
    """Inicializa el modelo de lenguaje una sola vez."""
    api_key = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = api_key
    return ChatGroq(
        #model="llama-3.3-70b-versatile",
        #model="gpt-oss-120b",  
        #model="qwen/qwen3.6-27b",
        model="openai/gpt-oss-120b",
        temperature=0,
        groq_api_key=api_key
    )

llm = inicializar_llm()

# 3. FUNCIÓN CRÍTICA: PROCESAMIENTO CACHEADO DEL PDF
@st.cache_resource
def procesar_documento(ruta_pdf):
    """Carga, fragmenta e indexa el PDF en un vector store persistido en caché."""
    loader = PyPDFLoader(ruta_pdf)
    docs = loader.load()
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    if not splits:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. INTERFAZ GRÁFICA (UI)
col1, col2 = st.columns([1, 2])
with col1:
    ruta_img = Path("images/img_ia.png")
    if ruta_img.exists():
        st.image(str(ruta_img), width=150)
with col2:
    st.header("Analizador de Documentos RAG (Retrieval-Augmented Generation)")
    st.write("🚀 Utiliza un agente de IA que es realmente sorprendente!")

with st.expander("Explicación del Programa"):
    st.write("""
        En las empresas se tienen cantidad de documentos, fichas de calidad, procedimientos, manuales, etc.
        Utilizando la IA se ahorra mucho tiempo con esta herramienta.\n
        **Ejemplos de preguntas simples:** \n
        - 'Muestra como limpiar el refrigerador'
        - 'Muestra la función power cool'
        - 'Muestra número de página de este contenido:...' \n
        *Nota de Seguridad:* Al utilizar la IA por seguridad no se recomendaría analizar documentos confidenciales en entornos nube. 
        Para máxima seguridad, esta misma herramienta puede correr 100% local (On-Premise) en una computadora moderna.
        """)

# 5. LÓGICA DE ARCHIVOS
archivos_pdf = [f for f in os.listdir('.') if f.endswith('.pdf')]

if not archivos_pdf:
    st.error("No se encontraron archivos pdf en la carpeta actual.")
else:
    archivo_seleccionado = st.selectbox("Selecciona el archivo que deseas analizar:", archivos_pdf)
    
    # El spinner y procesamiento SOLO ocurren si el archivo cambia gracias a @st.cache_resource
    with st.spinner("Procesando y preparando el documento dinámicamente..."):
        retriever = procesar_documento(archivo_seleccionado)
    
    if retriever is None:
        st.error("Error al procesar el documento. Asegúrate de que no esté vacío o dañado.")
        st.stop()

    # --- INTERFAZ DE USUARIO (Vista previa y Chat) ---
    with st.expander(f"Ver vista previa de {archivo_seleccionado}"):
        pdf_viewer(archivo_seleccionado)

    pregunta = st.chat_input("Haz una pregunta sobre los datos...")
    
    if pregunta:
      # 🛑 FILTRO DE SEGURIDAD (Anti-Prompt Injection)
        palabras_bloqueadas = ["environ", "secret", "os.", "sys.", "import", "open(", "write", "delete", "remove"]
    
        # Validar si la pregunta intenta acceder al sistema o variables
        if any(token in pregunta.lower() for token in palabras_bloqueadas):
            st.error("🛡️ Consulta bloqueada por políticas de seguridad del servidor. No se permiten comandos del sistema.")
        else:   
            st.info(f"🔍 **Consulta enviada:** {pregunta}")    
            with st.spinner("El agente está pensando y analizando los datos..."):
                try:
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

                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                    resultado = rag_chain.invoke({"input": pregunta})
                    
                    # Renderizado de resultados limpio
                    st.markdown(f"**Pregunta:** {pregunta}")
                    st.subheader("Respuesta del Asistente:")
                    st.success(resultado["answer"])

                except Exception as e:
                    st.error(f"Hubo un error al procesar la consulta: {e}")