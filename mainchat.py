from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Onde será salvo e a gestão dos dados aconteceram
from langchain_core.chat_history import InMemoryChatMessageHistory
# Onde esses dados poderão ser recuperados
from langchain_core.runnables.history import RunnableWithMessageHistory

set_debug(True)


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


modelo = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
)

# Placehouder armazenar historicos
prompt_sugestao = ChatPromptTemplate.from_messages([
    ("system", "Você é um guia de viagem especializado em destinos brasileiros. Apresente-se como Sr. Passeios"),
    ("placeholder", "{historico}"),
    ("human", "{query}")
])

cadeia = prompt_sugestao | modelo | StrOutputParser()

memoria = {}
sessao = "aula_langchain"

def historicoPorSessao(sessao : str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historicoPorSessao,
    input_messages_key="query",
    history_messages_key="historico"
)
        
lista_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

for uma_pergunta in lista_perguntas:
    resposta = cadeia_com_memoria.invoke(
        {
            "query" : uma_pergunta
        },
        config={"session_id":sessao}
    )
    print("Usuário: ", uma_pergunta),
    print("IA: ", resposta, "\n")