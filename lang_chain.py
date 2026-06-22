from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY, COHERE_API_KEY
from my_helper import encode_image

# Inicializar el modelo Gemini
llm = ChatGoogleGenerativeAI( 
    api_key=GEMINI_API_KEY,
    model=GEMINI_FLASH
)

# Inicializar el modelo Cohere
# llm_cohere = ChatCohere(
#     cohere_api_key=COHERE_API_KEY
# )

#Preguntas para ambos modelos
pregunta_texto = "¿Qué canales de YouTube recomiendas para aprender más sobre smartphones?"

# Consulta textual con Gemini
respuesta_texto = llm.invoke(pregunta_texto)
print(f"Respuesta Gemini: {respuesta_texto.content}\n")

# Consulta textual con Cohere
# respuesta_texto_cohere = llm_cohere.invoke(pregunta_texto)
# print(f"Respuesta Cohere: {respuesta_texto_cohere.content}\n")




# Convertir la imagen a base64
imagen = encode_image('datos/ejemplo_grafico.jpg')

pregunta = "Describe la imagen:"

mensaje = HumanMessage(
    content = [
        {
            "type": "text",
            "text": pregunta
        },
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{imagen}"
        }
    ]
)

# Enviar el mensaje multimodal
respuesta = llm.invoke([mensaje])

#Mostrar el resultado
print("Descripción de la imagen:")
print(respuesta)