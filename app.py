import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

# Variáveis de ambiente do arquivo .env
load_dotenv()

# Definindo suas chaves de API
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

# Definindo o modelo LLM, usando GPT-4
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o")

# Definindo a ferramenta de busca
search_tool = SerperDevTool(api_key=serper_api_key)

# Definindo os agentes com: funçoess e objetivos
analista_fundamentalista = Agent(
    role='Analista Financeiro Fundamentalista',
    goal='Analisar os fundamentos financeiros de BBAS3',
    backstory="""Você é um analista financeiro Senior especializado em análise fundamentalista.
    Seu trabalho é avaliar o valor intrínseco das empresas com base em seus balanços, demonstrações de resultados e outros indicadores financeiros.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[search_tool]
)

analista_tecnico = Agent(
    role='Analista Técnico de Ações',
    goal='Analisar os padrões gráficos e indicadores técnicos de BBAS3',
    backstory="""Você é um analista técnico Senior especializado em análise de gráficos de ações.
    Seu trabalho é identificar padrões de preço e volume, além de utilizar indicadores técnicos para prever movimentos futuros das ações.""",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools=[search_tool]
)

# Criando as tarefas dos agentes
tarefa_fundamentalista = Task(
    description="""Conduza uma análise fundamentalista completa da ação BBAS3.
    Avalie os balanços financeiros, demonstrações de resultados, fluxo de caixa, indicadores de rentabilidade, liquidez e endividamento.""",
    expected_output="Relatório completo com avaliação dos fundamentos financeiros da BBAS3",
    agent=analista_fundamentalista
)

tarefa_tecnica = Task(
    description="""Conduza uma análise técnica completa da ação BBAS3.
    Avalie os padrões gráficos, indicadores técnicos (como médias móveis, RSI, MACD) e identifique possíveis pontos de entrada e saída.""",
    expected_output="Relatório completo com avaliação técnica da BBAS3",
    agent=analista_tecnico
)

# Instanciando a equipe com um processo sequencial
equipe = Crew(
    agents=[analista_fundamentalista, analista_tecnico],
    tasks=[tarefa_fundamentalista, tarefa_tecnica],
    verbose=2,  # Você pode ajustar para 1 ou 2 para diferentes níveis de log
    process=Process.sequential
)

# Resultado da equipe
resultado = equipe.kickoff(inputs={"input": "BBAS3"})

print("######################")
print(resultado)
