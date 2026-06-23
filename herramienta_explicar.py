from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from my_keys import GEMINI_API_KEY
from my_models import GEMINI_FLASH
import ast

class HerramientaExplicar(BaseTool):
    name: str = "HerramientaExplicar"
    description: str = """
    Utiliza esta herramienta siempre que sea solicitada la explicacion de un 
    contenido a las personas.

    #ENTRADAS REQUERIDAS
    - 'tema'(str): Tema principal informado en la pregunta del usuario.
    """

    return_direct: bool = True

    def _run(self, accion):
        accion = ast.literal_eval(accion)
        tema_parametro = accion.get("tema", "")

        # Inicializar el modelo Gemini
        llm = ChatGoogleGenerativeAI( 
            api_key=GEMINI_API_KEY,
            model=GEMINI_FLASH
        )

        template_respuesta = PromptTemplate(
            template="""
            Asume el papel de un profesor con aspectos de didactica del usuario.

            1. Elabora una explicacion sobre el tema {tema} que sea de facil comprension para
            estudiantes de secundaria.
            2. Utiliza ejemplos cotidianos para volver la explicacion mas sencilla.
            3. En caso de que surja algun recurso para apoyar la explicacion, recuerda el escenario
            del contexto hispanoamericano.
            4. En caso que presentes algun script de codigo, se didactico y utiliza Python.

            tema pregunta: {tema}
            """,
            input_variables=["tema"]
        )

        cadena = template_respuesta | llm | StrOutputParser()

        respuesta = cadena.invoke({"tema": tema_parametro})
        
        return respuesta
