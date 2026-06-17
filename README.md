Autor: Gerardo Figueroa
Fecha: 10/06/26
# 🤖 Analizador de Documentos RAG Inteligente

Una solución corporativa e intuitiva de **Generación Aumentada por Recuperación (RAG)** construida con **Streamlit** y **LangChain**. Esta aplicación permite a los usuarios interactuar de manera conversacional con documentos PDF locales (como fichas técnicas, manuales de calidad y procedimientos de operación de Mabe) utilizando el modelo de última generación `llama-3.3-70b-versatile` a través de la infraestructura de ultra-baja latencia de **Groq**.

## 🎯 Características Principales

* **⚡ Arquitectura RAG de Extremo a Extremo:** Carga, fragmenta, indexa y consulta PDFs de forma dinámica y local.
* **🧠 Agente LLM Avanzado:** Potenciado por el modelo `llama-3.3-70b-versatile` a través de Groq Cloud para respuestas certeras y veloces.
* **📁 Procesamiento de Documentos bajo demanda:** Soporte integrado para manuales de la raíz del proyecto (ej. `mabe_polaris.pdf`, `mabe_rma.pdf`).
* **🗺️ Embeddings Locales Eficientes:** Uso de `sentence-transformers/all-MiniLM-L6-v2` mediante HuggingFace para la vectorización precisa de texto.
* **🗄️ Vector Store de Alta Velocidad:** Base de datos vectorial en memoria utilizando `Chroma` para de coincidencia de contexto optimizadas ($k=3$).
* **👁️ Visor de PDF Integrado:** Interfaz enriquecida con `streamlit-pdf-viewer` para contrastar las respuestas con el documento original en tiempo real.

## 🏗️ Estructura del Proyecto

La disposición de archivos en el repositorio sigue un patrón limpio y modular:
```text
.
├── images/
│   └── img_ia.png          # Logotipo e identidad visual de la IA en la interfaz
├── .streamlit/
│   └── secrets.toml        # Configuración local de credenciales (Ignorado en Git)
├── app.py                  # Código fuente principal de la aplicación Streamlit
├── mabe_polaris.pdf        # Manual de referencia técnica de ejemplo
├── mabe_rma.pdf            # Documento de procedimientos operativos de ejemplo
└── requirements.txt        # Definición de dependencias y librerías del sistema

## 🛠️ Requisitos Previos e Instalación
Siga estos pasos estructurados para desplegar y ejecutar el entorno de desarrollo local:

1. Clonar el Repositorio
Bash
git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
cd tu-repositorio
2. Configurar el Entorno Virtual
Bash
pip install --upgrade pip
pip install -r requirements.txt
Nota sobre requirements.txt: Asegúrese de contar con las librerías base especificadas en el código: streamlit, streamlit-pdf-viewer, langchain-groq, langchain-community, langchain-text-splitters, langchain-chroma, langchain-huggingface y chromadb.

## 🔑 Configuración de Credenciales y Variables de Entorno
La aplicación hace uso del sistema nativo de gestión de secretos de Streamlit para manejar de forma segura la API Key de Groq.
Cree un directorio llamado .streamlit en la raíz de su proyecto si no existe.
Dentro de esa carpeta, cree un archivo llamado secrets.toml.
Añada su clave de API de Groq de la siguiente manera:
Ini, TOML
# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_tu_clave_secreta_aqui_generada_en_groq_console"
⚠️ Importante: Jamás suba el archivo secrets.toml a su repositorio público de GitHub. Este archivo ya se encuentra configurado para ser ignorado en producción.

## 🚀 Ejecución de la Aplicación
Para inicializar el servidor web local de Streamlit y comenzar a interactuar con el agente RAG, ejecute el siguiente comando en la raíz del proyecto:
Bash
streamlit run app.py
Una vez ejecutado, se abrirá automáticamente una ventana en su navegador predeterminado apuntando a:

Local URL: http://localhost:8501

## 💡 Flujo de Trabajo e Instrucciones de Uso
Selección de Archivo: En el menú desplegable principal, elija uno de los documentos .pdf disponibles en la raíz (ej. mabe_polaris.pdf). El backend cargará e indexará el documento generando fragmentos de 1,000 caracteres con un solapamiento de 200 para no perder el contexto semántico.

Consulta de Datos: Utilice la barra de chat (chat_input) ubicada en la parte inferior para realizar consultas precisas.
Ejemplos de Preguntas:
“Muestra cómo limpiar el refrigerador”
“Muestra la función power cool”
“Muestra el número de página de este contenido: [...]”
Validación de Fuentes: Expanda el módulo “Ver vista previa de [archivo]” para hojear el documento original de manera interactiva sin salir de la app.

## 🔒 Nota de Seguridad y Despliegue On-Premise
Entornos Públicos: Al interactuar con APIs en la nube (como Groq Cloud), la información de los fragmentos recuperados viaja cifrada por HTTPS para ser procesada por el LLM. Por políticas corporativas de cumplimiento normativo, evite cargar información altamente confidencial o regulada por secretos comerciales en APIs de terceros.
Despliegue On-Premise (Local): Esta arquitectura es modular. Si requiere un entorno de confidencialidad absoluta (aire cerrado), la cadena LangChain permite sustituir ChatGroq por una instancia de Ollama ejecutando modelos locales como llama3 o mistral directamente en servidores internos o computadoras modernas con GPU dedicada, asegurando que ningún dato salga a internet.

🛠️ Tecnologías Utilizadas
Streamlit - Framework frontend ágil para aplicaciones de Machine Learning y Data Science.
LangChain (Classic & Core) - Orquestador para las cadenas de recuperación semántica y estructuración de prompts.
Groq Cloud API - Motor de inferencia ultrarrápido que da vida al modelo Llama.
ChromaDB - Almacenamiento e indexación vectorial eficiente de alto rendimiento.
Hugging Face Transformers - Generación de representaciones vectoriales densas de texto de alta calidad.

### 🌟 Aspectos destacados de este diseño de documentación:
1. **Badges de estado:** Añaden dinamismo visual a la cabecera (Python, Streamlit, Groq, LangChain).
2. **Sección de seguridad corporativa:** Resalta el valor del código (explicando la opción on-premise que mencionabas en tu `st.expander`).
3. **Instrucciones unificadas:** Bloques de código listos para copiar y pegar para Mac, Linux
