from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel
from dotenv import load_dotenv
from rich import print
import os

# Buscando a chave da Gemini API no arquivo .env
load_dotenv()
API_KEY=os.getenv("GEMINI_API_KEY")

# Inicializando o modelo Gemini 2.5 flash da Google
llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY)

# Criando a classe de estado para o StateGraph para processar as mensagens
class State(BaseModel):
    input: str
    output: str

# Criando função para processar a mensagem de entrada e gerar uma resposta usando o modelo Gemini
def responde_mensagem(state):
    input_mensagem = state.input
    resposta_llm = llm_model.invoke([HumanMessage(content=input_mensagem)])
    return State(input=state.input, output=resposta_llm.content)

# Criando o StateGraph para conseguir gerenciar os fluxos de mensagens que chegam
graph = StateGraph(State)

# Criando o nó no grafo para a função de resposta
graph.add_node("responde_mensagem", responde_mensagem)

# Definindo os pontos de entrada e saída do grafo
graph.set_entry_point("responde_mensagem")
graph.set_finish_point("responde_mensagem")

# Compilando o grafo para usa-lo
graph_compilado = graph.compile()

# Testando o grafo compilado com uma mensagem de entrada fixa
if __name__ == "__main__":
    resultado = graph_compilado.invoke(State(input="Olá, tudo bem com você?", output=""))
    print(resultado)
    
    # Gerar a visualização do grafo em formato Mermaid
    #print(graph_compilado.get_graph().draw_mermaid())
    