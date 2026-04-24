from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

import asyncio

from typing import Literal, TypedDict


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
)

prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sra Praia. Você é uma especialista em viagens com destinos para praia."),
        ("human", "{query}")
    ]
)

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sr Montanha. Você é uma especialista em viagens com destinos para montanhas e atividades radicais."),
        ("human", "{query}")
    ]
)

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()

cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

class Rota(TypedDict):
  destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda apenas com 'praia' ou 'montanha'"),
        ("human", "{query}")
    ]
)

roteador = prompt_roteador | modelo.with_structured_output(Rota)

class Estado(TypedDict):
  query:str
  destino:Rota
  resposta:str

# Chamadas para a IA um referente a cada nó e quais serão o retornos
async def no_roteador(estado: Estado, config=RunnableConfig):
    return {"destino" : await roteador.ainvoke({"query":estado["query"]}, config)}
async def no_praia(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_praia.ainvoke({"query": estado["query"]}, config)}
async def no_montanha(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_montanha.ainvoke({"query": estado["query"]}, config)}
def escolher_no(estado: Estado) -> Literal["praia", "montanha"]:
    return "praia" if estado["destino"]["destino"] == "praia" else "montanha"

grafo = StateGraph(Estado)

# "Nos" disponpiveis 
grafo.add_node("rotear", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)

# "Nos" indicado para iniciar o processo
grafo.add_edge(START, "rotear")

# "Nos" indicado desviar conforme o resultado
grafo.add_conditional_edges("rotear", escolher_no)

# "Nos" Finalizadores
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
    resposta = await app.ainvoke({"query": "Quero visitar um lugar no Brasil famoso por praias e cultura"})
    print(resposta["resposta"])

asyncio.run(main())