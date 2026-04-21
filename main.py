from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

numero_dias = 4
numero_criancas= 1
atividade="contemplar a natureza"

# Configuração do modelo, aqui poderia ser qualquer modelo que exista dentro do langchain para uso dos contructors
modelo = ChatOpenAI(
  model="gpt-3.5-turbo",
  temperature=0.5,
  api_key=api_key
)


prompt_cidade = PromptTemplate(
  """
  Sugira uma cidade dado o meu interesse por {interesse}.
  """,
  input_variables=["interrese"]
)

# Criação de cadeias, é onde a configuração realmente acontecerá, utilizando dos modelos e regras de prompts que foram sugeridas
cadeia = prompt_cidade | modelo | StrOutputParser

modelo = ChatOpenAI(
  model="gpt-3.5-turbo",
  temperature=0.5,
  api_key=api_key
)

resposta = cadeia.invoke({
  "interesse" : "praias"
})

print(resposta.content)



# Langchain tem um formato especifico para gerar a linguagem principal dos dados para gerar as cadeias de processos para que o fluxo da IA faça chamada especificas

# A cadeia depende de três elementos: 1-Estrutura do promp que vai ser utilizado; 2-Modelo de LLM que será utilizado; 3-Formato que espero receber

# Ter em mente o dormaro de saida é importe para ter o controle legitimo da saida dos dados

