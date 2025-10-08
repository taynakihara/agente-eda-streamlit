import streamlit as st
import pandas as pd
import numpy as np
import time
from utils.memoria_db import salvar_memoria, carregar_memoria

# Tentamos importar as libs de LLM, mas sem quebrar se não estiverem instaladas
try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ==========================================
# 🔹 Função auxiliar: resumo do dataset (com cache)
# ==========================================
@st.cache_data(show_spinner=False)
def summarize_dataset(df: pd.DataFrame) -> str:
    """Resumo básico do dataset (linhas, colunas, tipos, estatísticas)."""
    if df is None or df.empty:
        return "Nenhum dado foi carregado."

    resumo = [f"O dataset possui {df.shape[0]} linhas e {df.shape[1]} colunas."]
    tipos = df.dtypes.value_counts().to_dict()
    tipos_texto = ", ".join([f"{k}: {v}" for k, v in tipos.items()])
    resumo.append(f"Tipos de dados → {tipos_texto}.")

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
# 🔹 Memória de Chat
# ==========================================
def initialize_memory():
    """Inicializa a memória local e carrega as últimas conversas persistidas."""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "dataset_summary" not in st.session_state:
        st.session_state["dataset_summary"] = None

    # Carregar histórico persistente
    try:
        memoria_salva = carregar_memoria(limit=5)
        if memoria_salva:
            for item in memoria_salva[::-1]:  # ordem cronológica
                st.session_state["chat_history"].append(
                    {
                        "role": "assistant",
                        "content": f"🕒 {item['timestamp']}\n**{item['pergunta']}**\n\n{item['resposta']}",
                    }
                )
    except Exception as e:
        st.warning(f"⚠️ Não foi possível carregar memória persistente: {e}")


def add_to_history(role: str, content: str):
    st.session_state["chat_history"].append({"role": role, "content": content})


def show_history():
    """Exibe o histórico do chat."""
    for message in st.session_state["chat_history"]:
        st.chat_message(message["role"]).markdown(message["content"])


# ==========================================
# 🔹 Função de resposta do agente
# ==========================================
def generate_response(
    prompt: str, dataset_summary: str, api_key: str, provider: str
) -> str:
    """Gera resposta real ou simulada de acordo com o provider."""
    if not api_key or not provider:
        return (
            f"💬 [Modo offline] Você perguntou: '{prompt}'\n\n{dataset_summary[:500]}"
        )

    # --- OPENAI ---
    if provider == "OpenAI" and openai:
        try:
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
        except Exception as e:
            return f"⚠️ Erro ao conectar à API OpenAI: {e}"

    # --- GROQ ---
    elif provider == "Groq" and openai:
        try:
            client = openai.OpenAI(
                api_key=api_key, base_url="https://api.groq.com/openai/v1"
            )
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
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
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Erro ao conectar à API Groq: {e}"

    # --- GEMINI ---
    elif provider == "Gemini" and genai:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(
                f"Usuário: {prompt}\n\nContexto:\n{dataset_summary[:1000]}"
            )
            return response.text
        except Exception as e:
            return f"⚠️ Erro ao conectar à API Gemini: {e}"

    return "⚠️ Nenhum provedor válido configurado ou biblioteca ausente."


# ==========================================
# 🔹 Função principal do Chat
# ==========================================
def render_chat(
    data,
    numeric_cols,
    categorical_cols,
    dataset_summary=None,
    api_key=None,
    provider=None,
):
    st.markdown("### 💬 Chat Interativo com Memória Persistente")
    initialize_memory()
    show_history()

    # Entrada do usuário
    user_input = st.chat_input("Digite sua pergunta sobre os dados...")
    if not user_input:
        return

    add_to_history("user", user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.info("🤖 Processando sua pergunta...")

        try:
            response = generate_response(user_input, dataset_summary, api_key, provider)
        except Exception as e:
            response = f"⚠️ Erro ao processar sua solicitação: {e}"
        finally:
            placeholder.empty()

        st.markdown(response)
        add_to_history("assistant", response)

        # 💾 Salvar pergunta e resposta na memória persistente Supabase
        try:
            salvar_memoria(user_input, response, tipo_analise="chat")
        except Exception as e:
            st.warning(f"⚠️ Não foi possível salvar na memória persistente: {e}")
