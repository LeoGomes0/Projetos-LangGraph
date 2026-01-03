from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel
from dotenv import load_dotenv
from rich import print

import os

# Buscando chave da Gemini API no arquivo .env
load_dotenv()
API_KEY=os.getenv("GEMINI_API_KEY")

# Inicializando o modelo Gemini 2.5 flash da Google
llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY)

# Criando a classe de estado para o StateGraph para processar mensagens
class State(BaseModel):
    input: str
    output: str

# Função para processar a mensagem de entrada e gerar uma resposta usando o modelo Gemini
def responde_mensagem(state):
    mensagem_entrada = state.input
    resposta_llm = llm_model.invoke([HumanMessage(content=mensagem_entrada)])
    return State(input=state.input, output=resposta_llm.content)

# Criando o StateGraph para gerenciar os fluxos de mensagens
graph = StateGraph(State)

# Adicionando a função de resposta como um nó no grafo
graph.add_node("responde_mensagem", responde_mensagem)

# Definindo os pontos de entrada e saída do grafo
graph.set_entry_point("responde_mensagem")
graph.set_finish_point("responde_mensagem")

# Compilando o grafo para usa-lo
export_graph = graph.compile()

# Um exemplo de uso do grafo para processar uma mensagem
if __name__ == "__main__":
    resultado = export_graph.invoke(State(input="Quero que se apresente em poucas palavras", output=""))
    print(resultado)

