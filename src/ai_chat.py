import streamlit as st
import pandas as pd
import numpy as np
import concurrent.futures
from utils.memoria_db import salvar_memoria, carregar_memoria

# Tentativa de import das bibliotecas de LLM
try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ==========================================
# 🔹 Resumo de dataset (com cache)
# ==========================================
@st.cache_data(show_spinner=False)
def summarize_dataset(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "Nenhum dado foi carregado."

    resumo = [f"O dataset possui {df.shape[0]} linhas e {df.shape[1]} colunas."]
    tipos = df.dtypes.value_counts().to_dict()
    resumo.append(
        "Tipos de dados → " + ", ".join([f"{k}: {v}" for k, v in tipos.items()])
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        stats = df[numeric_cols].describe().T[["mean", "std", "min", "max"]]
        resumo.append("Estatísticas resumidas das variáveis numéricas:")
        for col, row in stats.iterrows():
            resumo.append(
                f"• {col}: média={row['mean']:.2f}, desvio={row['std']:.2f}, "
                f"min={row['min']:.2f}, max={row['max']:.2f}"
            )
    return "\n".join(resumo)


# ==========================================
# 🧠 Memória Persistente
# ==========================================
def initialize_memory():
    if "chat_history" not in st.session_state:
        # Inicializa se for a primeira vez
        st.session_state["chat_history"] = []


def add_to_history(role, content):
    st.session_state["chat_history"].append({"role": role, "content": content})


def show_history():
    for message in st.session_state["chat_history"]:
        st.chat_message(message["role"]).markdown(message["content"])


# ==========================================
# 🤖 Geração de resposta
# ==========================================
def generate_response(
    prompt, chat_history, dataset_summary, api_key, provider, model_name=None
):
    if not api_key or not provider:
        return "⚠️ Configure o provedor e insira a chave da API antes de usar o chat."

    # Novo: Prepara o histórico para a API
    system_prompt = "Você é um analista de dados útil e explicativo. Seu conhecimento é limitado ao contexto do dataset fornecido."
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Contexto do Dataset: {dataset_summary[:1000]}",
        },  # Adiciona o contexto como a primeira mensagem do usuário/sistema
    ]
    # 2. Histórico da conversa (limpando o timestamp da memória)
    for msg in chat_history[-6:]:  # Limita o histórico para as últimas 6 mensagens
        content = msg["content"]
        if content.startswith("🕒"):
            content = "\n".join(content.split("\n")[2:])
        # Tenta pegar apenas o conteúdo após o timestamp
        messages.append({"role": msg["role"], "content": content.strip()})

    # 3. Adiciona o prompt atual do usuário como a última mensagem
    messages.append({"role": "user", "content": prompt})

    try:
        if provider == "OpenAI" and openai:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                # USAR A NOVA LISTA DE MENSAGENS
                messages=messages,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        elif provider == "Groq" and openai:
            client = openai.OpenAI(
                api_key=api_key, base_url="https://api.groq.com/openai/v1"
            )
            model_to_use = model_name or "llama3-8b-8192"
            response = client.chat.completions.create(
                model=model_to_use,
                # USAR A NOVA LISTA DE MENSAGENS
                messages=messages,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        elif provider == "Gemini" and genai:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            # PRECISA USAR O CHAT SERVICE DO GEMINI PARA MANTER A MEMÓRIA
            # Vamos simular a passagem de contexto no prompt por simplicidade AGORA, mas
            # o ideal seria usar o client.chats().send_message() para Gemini.
            context_prompt = "\n".join(
                [f"{m['role']}: {m['content']}" for m in messages]
            )
            response = model.generate_content(context_prompt)
            return response.text

        else:
            return "⚠️ Nenhum provedor válido configurado ou biblioteca ausente."

    # Trata erros de cota e conexão de forma mais amigável
    except Exception as e:
        erro_str = str(e)

        if "insufficient_quota" in erro_str or "429" in erro_str:
            return "⚠️ Erro de cota/limite de uso excedido. Verifique seu plano na API do provedor."
        elif "API key is not valid" in erro_str or "401" in erro_str:
            return "⚠️ Erro de autenticação. A API Key inserida é inválida."

        return f"⚠️ Erro ao conectar à API: {e}"


# ==========================================
# ⚙️ Execução Assíncrona
# ==========================================
def generate_response_async(*args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(generate_response, *args, **kwargs)
        return future.result()


# ==========================================
# 💬 Chat com spinner seguro
# ==========================================
def render_chat(
    data,
    numeric_cols,
    categorical_cols,
    dataset_summary=None,
    api_key=None,
    provider=None,
):
    """Chat IA com spinner leve e seguro (sem overlay bloqueante)."""
    initialize_memory()

    if not provider or not api_key:
        st.warning("⚠️ Configure o provedor e a API Key para usar o chat.")
        return

    show_history()
    user_input = st.chat_input("Digite sua pergunta sobre os dados...")
    if not user_input:
        return

    add_to_history("user", user_input)

    # Spinner embutido no chat (sem overlay global)
    with st.chat_message("assistant"):
        with st.spinner("🤖 Processando sua pergunta com IA..."):
            try:
                # ❗ APENAS ESTE BLOCO É ALTERADO PARA INCLUIR O HISTÓRICO
                resposta = generate_response_async(
                    user_input,
                    st.session_state["chat_history"],  # <-- NOVO: Histórico do chat
                    dataset_summary,
                    api_key,
                    provider,
                    model_name=st.session_state.get(
                        "groq_model", "llama-3.2-8b-text-preview"
                    ),
                )
            except Exception as e:
                resposta = f"⚠️ Erro: {e}"

            st.markdown(resposta)
            add_to_history("assistant", resposta)

        try:
            salvar_memoria(user_input, resposta, tipo_analise="chat")
        except Exception as e:
            st.warning(f"⚠️ Erro ao salvar na memória: {e}")
