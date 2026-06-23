from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY, COHERE_API_KEY
from my_helper import encode_image
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug

set_debug(True)

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
# pregunta_texto = "¿Qué canales de YouTube recomiendas para aprender más sobre smartphones?"

# Consulta textual con Gemini
# respuesta_texto = llm.invoke(pregunta_texto)
# print(f"Respuesta Gemini: {respuesta_texto.content}\n")

# Consulta textual con Cohere
# respuesta_texto_cohere = llm_cohere.invoke(pregunta_texto)
# print(f"Respuesta Cohere: {respuesta_texto_cohere.content}\n")




# Convertir la imagen a base64
imagen = encode_image('datos/ejemplo_grafico.jpg')

pregunta = "Describe la imagen:"

template_analisis = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
         Asume que eres analista de imágenes. Tu principal tarea consiste en: analizar una imagen para 
         extraer las informaciones más relevantes de manera objetiva.

         #FORMATO DE SALIDA
         Descripción de la imagen: Tu descripción de la imagen aqui.
         Etiquetas: Una lista con 3 palabras clave separadas por comas.
         """
        ),
        (
            "user", 
            [
                {
                    "type": "text",
                    "text": "Describe la imagen:"
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{imagen_informada}"
                }
            ]
        )
    ]
)

# Enviar el mensaje multimodal
cadena_analisis = template_analisis | llm | StrOutputParser()

respuesta_analisis = cadena_analisis.invoke({"imagen_informada": imagen})

#Mostrar el resultado
print("Descripción de la imagen:")
print(respuesta_analisis)



#Otra cadena para generar un resumen de la respuesta del analisis de la imagen
template_respuesta = PromptTemplate(
    template="""
            Genera un resumen, utlizando un lenguaje claro y objetivo, enfocado en el publico cololombiano.
            La idea es que la comunicación del resultado sea lo mas sencilla posible, priorizando los registros
            para consultas posteriores.

            #RESULTADO DE LA IMAGEN
            {respuesta_analisis_imagen}
            """,
            input_variables=["respuesta_analisis_imagen"]
)

cadena_resumen = template_respuesta | llm | StrOutputParser()

cadena_compuesta = (cadena_analisis | cadena_resumen)

respuesta = cadena_compuesta.invoke({"imagen_informada": imagen})