from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.globals import set_debug

set_debug(True)

# Lib em pyton para "tipar" os dados que esperado de saida, o interface é o equivalente ao base model
from pydantic import Field, BaseModel

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Extende o Base model informando os tipos que serão usado, para gerar um parce mais estruturado dos dados enviados
class Destino(BaseModel):
  cidade: str = Field("A cidade recomendada para visitar")
  motivo: str =  Field("O Motivo da recomendação de cidade")

# Extende as informações do parseador incluindo dentro dele os dados da Tipagem criada
parseador = JsonOutputParser(pydantic_object=Destino)
  

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
  {formato_de_saida}
  """,
  input_variables=["interrese"],
  partial_variables={
    "formato_de_saida" :parseador.get_format_instructions()
  }
)

# Criação de cadeias, é onde a configuração realmente acontecerá, utilizando dos modelos e regras de prompts que foram sugeridas
cadeia = prompt_cidade | modelo | parseador

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

