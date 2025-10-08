# Análise Exploratória de Dados com IA

Aplicação Streamlit para análise completa de arquivos CSV com chat inteligente.

## Funcionalidades

- **Distribuições**: Exibe histogramas e gráficos de barras para mostrar a distribuição e frequência das variáveis numéricas e categóricas do dataset.
- **Correlações**: Gera uma matriz de correlação completa, destacando relações estatísticas entre variáveis e possíveis dependências.
- **Tendências**: Analisa o comportamento temporal dos dados, identificando padrões, sazonalidades e movimentos de tendência.
- **Variância**: Calcula e visualiza a variação das variáveis, ajudando a entender a dispersão e relevância de cada atributo.
- **Anomalias**: Detecta e destaca outliers com métodos estatísticos (IQR e Z-score), auxiliando na identificação de inconsistências.
- **Clusters**: Realiza agrupamento automático de dados semelhantes (clustering), revelando padrões ocultos e segmentos naturais no dataset
- **Chat IA**: Permite interação em linguagem natural com os dados carregados, oferecendo explicações e insights personalizados com memória contextual.

## APIs Suportadas

- **OpenAI**: GPT-3.5, GPT-4
- **Groq**: Llama 3.3 70B, Llama 3.1 8B
- **Gemini**: Gemini Pro

## Como Usar

1. **Instalar dependências:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Executar aplicação:**

   ```bash
   streamlit run app.py
   ```

3. **Usar a aplicação:**
   - Carregue arquivo CSV
   - Explore as análises automáticas
   - Configure API de IA e faça perguntas sobre os dados

## Deploy no Streamlit Cloud

1. Faça upload dos 3 arquivos para repositório GitHub
2. Conecte repositório no [share.streamlit.io](https://share.streamlit.io)
3. Deploy automático

## Chaves das APIs

- **OpenAI**: [platform.openai.com](https://platform.openai.com) → API Keys
- **Groq**: [console.groq.com](https://console.groq.com) → API Keys
- **Gemini**: [makersuite.google.com](https://makersuite.google.com) → Get API Key
