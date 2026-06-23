from pydantic import BaseModel, Field
from typing import List

class DetallesImagen(BaseModel):
    titulo:str = Field(
        description = "Define el titulo adecuado para la imagen analizada."
    )
    descripcion:str = Field(
        description="Coloca aqui una descripcion detallada del analisis de la imagen."
    )
    etiquetas:List[str] = Field(
        description="Define 3 palabras-clave para la imagen analizada"
    )