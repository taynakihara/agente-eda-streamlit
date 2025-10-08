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
        st.session_state["chat_history"] = []
    if "dataset_summary" not in st.session_state:
        st.session_state["dataset_summary"] = None

    if st.session_state.get("memoria_carregada"):
        return

    try:
        memoria_salva = carregar_memoria(limit=5)
        if memoria_salva:
            st.session_state["chat_history"] = [
                {
                    "role": "assistant",
                    "content": f"🕒 {item['timestamp']}\n**{item['pergunta']}**\n\n{item['resposta']}",
                }
                for item in memoria_salva[::-1]
            ]
        st.session_state["memoria_carregada"] = True
    except Exception as e:
        st.warning(f"⚠️ Não foi possível carregar memória persistente: {e}")


def add_to_history(role, content):
    st.session_state["chat_history"].append({"role": role, "content": content})


def show_history():
    for message in st.session_state["chat_history"]:
        st.chat_message(message["role"]).markdown(message["content"])


# ==========================================
# 🤖 Geração de resposta
# ==========================================
def generate_response(prompt, dataset_summary, api_key, provider, model_name=None):
    if not api_key or not provider:
        return "⚠️ Configure o provedor e insira a chave da API antes de usar o chat."

    try:
        if provider == "OpenAI" and openai:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um analista de dados útil e explicativo.",
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n{dataset_summary[:1000]}",
                    },
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        elif provider == "Groq" and openai:
            if not api_key.startswith("gsk_"):
                return "⚠️ Chave da API Groq inválida. Ela deve começar com 'gsk_'."
            client = openai.OpenAI(
                api_key=api_key, base_url="https://api.groq.com/openai/v1"
            )
            model_to_use = model_name or "llama-3.2-8b-text-preview"
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um analista de dados especializado em EDA.",
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n{dataset_summary[:1000]}",
                    },
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        elif provider == "Gemini" and genai:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                f"Usuário: {prompt}\n\nContexto:\n{dataset_summary[:1000]}"
            )
            return response.text

        else:
            return "⚠️ Nenhum provedor válido configurado ou biblioteca ausente."

    except Exception as e:
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
    st.markdown("### 💬 Chat Interativo com Memória Persistente")
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
                resposta = generate_response_async(
                    user_input,
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
