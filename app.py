import streamlit as st
from src.data_loader import load_data
from src.analysis import (
    distributions,
    correlations,
    trends,
    anomalies,
    clustering,
    variance,
)
from src.ai_chat import render_chat, summarize_dataset
from utils.cache_utils import cache_clear_button
from dotenv import load_dotenv
import pandas as pd

# ====================================================
# 🔧 Configurações Iniciais
# ====================================================
load_dotenv()
st.set_page_config(page_title="🤖 Análise Exploratória com IA", layout="wide")

# ====================================================
# 💅 Estilos Globais
# ====================================================
st.markdown(
    """
    <style>
        .block-container { margin-top: 3.5rem; }
        header, .stAppHeader { margin-top: 0 !important; padding-top: 0 !important; }
        .main { padding-bottom: 2rem; }
        body, .stApp {
            background: linear-gradient(180deg, #001F3F, #003366, #004080, #0059b3);
            color: white;
        }
        .block-container { max-width: 900px; margin: 0 auto; padding-top: 2rem; }
        section.main > div { display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .stFileUploader { width: 70%; max-width: 600px; margin: 0 auto; }
        .stAlert { max-width: 600px; margin: 1rem auto; border-radius: 10px; font-size: 0.9rem; }
        h1 { text-align: center !important; font-size: 2.4rem !important; }
        h2, h3 { text-align: center; color: #A7C7E7; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ====================================================
# 📤 Cabeçalho e Upload
# ====================================================
st.title("📊 Análise Exploratória de Dados com IA")
uploaded_file = st.file_uploader("📂 Envie seu arquivo CSV para análise", type=["csv"])

# ====================================================
# 🎨 Ajuste visual dinâmico (após upload)
# ====================================================
if uploaded_file:
    st.markdown(
        """
        <style>
            .block-container { max-width: 95% !important; padding-left: 3%; padding-right: 3%; }
            .stPlotlyChart, .stPyplot { width: 100% !important; max-width: 100% !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ====================================================
# ⚙️ Lógica Principal
# ====================================================
if uploaded_file:
    # Exibe loading inicial
    loading_container = st.empty()
    loading_container.markdown(
        """
        <div id="loading-overlay">
            <div class="loading-content">
                <div class="spinner"></div>
                <p>Carregando gráficos e análises... aguarde ⏳</p>
            </div>
        </div>
        <style>
            #loading-overlay {
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background-color: rgba(0, 0, 30, 0.97);
                display: flex; justify-content: center; align-items: center;
                z-index: 9999; flex-direction: column; color: #A7C7E7;
                font-size: 1.2rem; text-align: center;
            }
            .spinner {
                border: 6px solid rgba(255, 255, 255, 0.2);
                border-top: 6px solid #66B2FF;
                border-radius: 50%;
                width: 70px; height: 70px;
                animation: spin 1.2s linear infinite;
                margin-bottom: 20px;
            }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Carrega dados
    data, numeric_cols, categorical_cols = load_data(uploaded_file)
    st.success(
        f"✅ Arquivo carregado: {data.shape[0]} linhas, {data.shape[1]} colunas."
    )
    cache_clear_button()

    # ====================================================
    # 🧩 Criação das Abas
    # ====================================================
    tabs = st.tabs(
        [
            "📊 Distribuições",
            "🔍 Correlações",
            "📈 Tendências",
            "📉 Variância",
            "⚠️ Anomalias",
            "🧩 Clusters",
            "🤖 Chat IA",
        ]
    )

    # ====================================================
    # 📈 Renderização das Abas
    # ====================================================
    with tabs[0]:
        distributions.render(data, numeric_cols, categorical_cols)
    with tabs[1]:
        correlations.render(data, numeric_cols)
    with tabs[2]:
        trends.render(data, numeric_cols)
    with tabs[3]:
        variance.render(data, numeric_cols)
    with tabs[4]:
        anomalies.render(data, numeric_cols)
    with tabs[5]:
        clustering.render(data, numeric_cols)

    # ====================================================
    # 🤖 Aba do Chat IA (com memória persistente)
    # ====================================================
    with tabs[6]:
        st.header("🤖 Chat Inteligente com Memória Persistente")

        # Inicialização da memória
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Gera resumo do dataset (se necessário)
        if "dataset_summary" not in st.session_state:
            st.session_state["dataset_summary"] = summarize_dataset(data)

        dataset_summary = st.session_state["dataset_summary"]

        # Botão para limpar memória
        if st.button("🧹 Limpar memória do agente"):
            st.session_state["chat_history"] = []
            st.session_state["dataset_summary"] = None
            st.success("✅ Memória limpa com sucesso!")

        # ====================================================
        # 🔑 Configuração da API
        # ====================================================
        st.subheader("🔑 Configuração da API da IA")
        if "provider" not in st.session_state:
            st.session_state["provider"] = "OpenAI"
        if "user_api_key" not in st.session_state:
            st.session_state["user_api_key"] = ""
        if "groq_model" not in st.session_state:
            st.session_state["groq_model"] = "llama-3.2-8b-text-preview"

        provider = st.selectbox(
            "Selecione o provedor de IA:",
            ["OpenAI", "Groq", "Gemini"],
            index=["OpenAI", "Groq", "Gemini"].index(st.session_state["provider"]),
            key="provider_selector",
        )

        api_key = st.text_input(
            f"Insira sua API Key ({provider})",
            type="password",
            value=st.session_state["user_api_key"],
            key="user_api_key_input",
        )

        # Exibe opção de modelo apenas se for Groq
        if provider == "Groq":
            model_name = st.selectbox(
                "Selecione o modelo Groq:",
                ["llama-3.2-8b-text-preview", "llama-3.2-70b-text-preview"],
                index=["llama-3.2-8b-text-preview", "llama-3.2-70b-text-preview"].index(
                    st.session_state["groq_model"]
                ),
            )
            st.session_state["groq_model"] = model_name
        else:
            model_name = None

        # Botão para salvar configuração
        if st.button("💾 Salvar Configuração de API"):
            st.session_state["provider"] = provider
            st.session_state["user_api_key"] = api_key
            if provider == "Groq":
                st.session_state["groq_model"] = model_name
            st.success(f"✅ Configuração salva: {provider}")

        # Renderiza o chat
        render_chat(
            data=data,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            dataset_summary=dataset_summary,
            api_key=st.session_state.get("user_api_key"),
            provider=st.session_state.get("provider"),
        )

    # Remove overlay
    st.session_state["loaded"] = True
    loading_container.empty()

else:
    st.info("👆 Envie um arquivo CSV para começar a análise.")
