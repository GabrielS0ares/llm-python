from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from langchain_core.globals import set_debug

set_debug(True)

# Lib em pyton para "tipar" os dados que esperado de saida, o interface é o equivalente ao base model
from pydantic import Field, BaseModel

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Extende o Base model informando os tipos que serão usado, para gerar um parce mais estruturado dos dados enviados
class Destino(BaseModel):
  cidade: str = Field("A cidade recomendada para visitar")
  motivo: str =  Field("O Motivo da recomendação de cidade")

class Restaurantes(BaseModel):
  cidade: str = Field("A cidade recomendada para visitar")
  restaurante: str = Field("Restaurante indicados para visitar")

# Extende as informações do parseador incluindo dentro dele os dados da Tipagem criada
parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurante = JsonOutputParser(pydantic_object=Restaurantes)

# Configuração do modelo, aqui poderia ser qualquer modelo que exista dentro do langchain para uso dos contructors
modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
)


prompt_cidade = PromptTemplate(
  template="""
  Sugira uma cidade dado o meu interesse por {interesse}.
  {formato_de_saida}
  """,
  input_variables=["interrese"],
  partial_variables={
    "formato_de_saida" :parseador_destino.get_format_instructions()
  }
)

prompt_restaurante = PromptTemplate(
  template="""
  Sugira restaurantes populares entre locais {cidade}.
  {formato_de_saida}
  """,
  partial_variables={
    "formato_de_saida" :parseador_restaurante.get_format_instructions()
  }
)

prompt_cultural = PromptTemplate(
  template="Sugira atividades e locais culturais em {cidade}"
)


# Criação de cadeias, é onde a configuração realmente acontecerá, utilizando dos modelos e regras de prompts que foram sugeridas
cadeia_1 = prompt_cidade | modelo | parseador_destino
cadeia_2 = prompt_restaurante | modelo | parseador_restaurante
cadeia_3 = prompt_cultural | modelo | StrOutputParser()

# Criação da caia composta onde estará a conversação entre as cadeias informadas
cadeia = (cadeia_1 | cadeia_2 | cadeia_3)

resposta = cadeia.invoke({
  "interesse" : "praias"
})

print(resposta)



# Langchain tem um formato especifico para gerar a linguagem principal dos dados para gerar as cadeias de processos para que o fluxo da IA faça chamada especificas

# A cadeia depende de três elementos: 1-Estrutura do promp que vai ser utilizado; 2-Modelo de LLM que será utilizado; 3-Formato que espero receber

# Ter em mente o dormaro de saida é importe para ter o controle legitimo da saida dos dados

