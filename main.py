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

modelo_de_prompt = PromptTemplate(
  template="""
Crie um roteiro de viagem de {numero_dias} dias, para uma familia com {numero_criancas} de 1 ano, que gostam de {atividade}
"""
)

# Aqui é onde alimenta a estrutura que foi usada para limitar o uso de tokens, assim fechando o contexto do uso
prompt = modelo_de_prompt.format(
  numero_dias=numero_dias,
  numero_criancas=numero_criancas,
  atividade=atividade
)


# Configuração do modelo, aqui poderia ser qualquer modelo que exista dentro do langchain para uso dos contructors
modelo = ChatOpenAI(
  model="gpt-3.5-turbo",
  temperature=0.5,
  api_key=api_key
)

# Langchain tem um formato especifico para gerar a linguagem principal dos dados para gerar as cadeias de processos para que o fluxo da IA faça chamada especificas

# A cadeia depende de três elementos: 1-Estrutura do promp que vai ser utilizado; 2-Modelo de LLM que será utilizado; 3-Formato que espero receber

# Ter em mente o dormaro de saida é importe para ter o controle legitimo da saida dos dados


resposta = modelo.invoke(prompt)

print(resposta.content)

print(resposta)