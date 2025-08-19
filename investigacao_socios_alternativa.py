# Iniciando a automacao

import pandas as pd # type: ignore
import os
import google.generativeai as genai # type: ignore
from dotenv import load_dotenv # type: ignore
import re

# ----------------------------------------
# Parte 1: Configurar a API do Gemini
# ----------------------------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Erro: A chave de API do Gemini não foi encontrada.")
    exit()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# ----------------------------------------
# Parte 2: Carregar e Analisar o Arquivo de Sócios
# ----------------------------------------
nome_arquivo_socios = 'socios.csv'

try:
    df_socios = pd.read_csv(
        nome_arquivo_socios,
        delimiter=';',
        encoding='ISO-8859-1',
        nrows=20000, # Carregamos 20 mil linhas para ter uma amostra grande
        header=None,
        dtype={0: str}
    )
    df_socios = df_socios.rename(columns={0: 'CNPJ_EMPRESA', 2: 'NOME_SOCIO'})
    df_socios['CNPJ_EMPRESA'] = df_socios['CNPJ_EMPRESA'].str.replace(r'\D', '', regex=True)
    df_socios['NOME_SOCIO'] = df_socios['NOME_SOCIO'].str.replace(r'[^a-zA-Z0-9\s]+', '', regex=True)
    df_socios['NOME_SOCIO'] = df_socios['NOME_SOCIO'].str.replace(r'[0-9]', '', regex=True).str.strip()


    socios_duplicados = df_socios[df_socios.duplicated(subset=['NOME_SOCIO'], keep=False)]

    if socios_duplicados.empty:
        print("\nNenhum sócio duplicado encontrado na amostra.")
        exit()

    print(f"\nSucesso! Encontrados {len(socios_duplicados)} sócios que se repetem em diferentes empresas.")
    print("\nAmostra dos sócios duplicados:")
    print(socios_duplicados[['CNPJ_EMPRESA', 'NOME_SOCIO']].drop_duplicates().head(10))

except FileNotFoundError:
    print("Erro: O arquivo de sócios não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao processar os dados: {e}")
    exit()

# ----------------------------------------
# Parte 3: Criar e Enviar o Prompt para o Gemini
# ----------------------------------------
dados_para_gemini = socios_duplicados[['CNPJ_EMPRESA', 'NOME_SOCIO']].drop_duplicates().head(10).to_string()

pergunta = "Analise a tabela de sócios e CNPJs. Existem sócios que se repetem em diferentes empresas? Se sim, quais são, e o que isso pode indicar? Há algum sócio que tenha BARBOSA ou PEIXOTO como parte do nome? Há alguma referência a palavra "MAXIMINIO"?"

prompt_completo = f"Tabela de sócios com CNPJs:\n\n{dados_para_gemini}\n\n{pergunta}"

print("\nEnviando a solicitação de análise de conexões para o Gemini...")

try:
    response = model.generate_content(prompt_completo)
    print("\n--- Resposta do Gemini ---")
    print(response.text)
except Exception as e:
    print(f"\nErro ao gerar a resposta do Gemini: {e}")
