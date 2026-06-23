from langchain.agents import AgentExecutor
from orquestador import AgenteOrquestador

def main():
    agente = AgenteOrquestador()
    ejecutor = AgentExecutor(
        agent=agente.agente, 
        tools=agente.tools,
        verbose=True
    )

    # pregunta = "Realiza el analisis de la imagen ejemplo_grafico.jpg"
    pregunta = "Quiero que me expliques como funcionan los desvios condicionales"

    respuesta = ejecutor.invoke({"input": pregunta})

    print(f"Respuesta del agente: {respuesta}\n")

if __name__ == "__main__":
    main()