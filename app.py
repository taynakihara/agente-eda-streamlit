import streamlit as st
from src.ai_chat import render_chat, summarize_dataset
from src.data_loader import load_data
from src.analysis import (
    distributions,
    correlations,
    trends,
    anomalies,
    clustering,
    variance,
)
from utils.cache_utils import cache_clear_button
from dotenv import load_dotenv
import pandas as pd

# ===============================
# 🔧 Configurações iniciais
# ===============================
load_dotenv()
st.set_page_config(page_title="🤖 Análise com IA", layout="wide")

# ===============================
# 🎨 Estilos globais (layout centralizado e menor)
# ===============================
st.markdown(
    """
    <style>
        /* Fundo em degradê */
        body, .stApp {
            background: linear-gradient(180deg, #001F3F, #003366, #004080, #0059b3);
            color: white;
        }

        /* Container principal centralizado */
        .block-container {
            max-width: 1300px;  /* deixa mais estreito */
            margin: 0 auto;
            padding-top: 2rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        /* Títulos centralizados */
        h1, h2, h3 {
            text-align: center !important;
            color: #A7C7E7 !important;
        }

        /* Caixa de upload mais compacta */
        .stFileUploader {
            width: 70% !important;
            max-width: 550px !important;
            margin: 1rem auto !important;
        }

        /* Caixa de informação centralizada */
        .stAlert {
            width: 70% !important;
            max-width: 550px !important;
            margin: 0 auto;
            text-align: center;
            border-radius: 10px;
        }

        /* Ajuste para barra superior */
        header, .stAppHeader {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# ===============================
# 📤 Upload do CSV
# ===============================
st.title("📊 Análise de Dados com IA")
uploaded_file = st.file_uploader("📂 Envie seu arquivo CSV", type=["csv"])

if uploaded_file:
    # ====================================================
    # 🌀 LOADING VISUAL EM TELA CHEIA (trava ações do usuário)
    # ====================================================
    loading_container = st.empty()
    loading_container.markdown(
        """
        <div id="loading-overlay">
            <div class="loading-content">
                <div class="spinner"></div>
                <p>Carregando dados e análises... aguarde ⏳</p>
            </div>
        </div>
        <style>
            #loading-overlay {
                position: fixed;
                top: 0; left: 0;
                width: 100%; height: 100%;
                background-color: rgba(0, 0, 30, 0.95);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                flex-direction: column;
                color: #A7C7E7;
                font-size: 1.2rem;
                text-align: center;
            }
            .spinner {
                border: 6px solid rgba(255, 255, 255, 0.2);
                border-top: 6px solid #66B2FF;
                border-radius: 50%;
                width: 70px; height: 70px;
                animation: spin 1.2s linear infinite;
                margin-bottom: 20px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Agora processa os dados normalmente
    data, numeric_cols, categorical_cols = load_data(uploaded_file)

    st.success(
        f"✅ Arquivo carregado: {data.shape[0]} linhas, {data.shape[1]} colunas."
    )

    def cache_clear_button():
        """Botão para limpar cache, sessão e memória do app completamente."""
        if st.button("🧹 Limpar Cache", key="clear_cache_button"):
            try:
                # 🔹 Limpa todos os caches do Streamlit
                st.cache_data.clear()
                st.cache_resource.clear()

                # 🔹 Limpa variáveis da sessão (sem quebrar o app)
                for key in list(st.session_state.keys()):
                    del st.session_state[key]

                # 🔹 Mostra mensagem de sucesso
                st.success(
                    "✅ Cache e sessão limpos com sucesso! Recarregue o arquivo CSV."
                )

                # 🔹 Força recarregamento da página
                st.rerun()

            except Exception as e:
                st.error(f"⚠️ Erro ao limpar cache: {e}")

    # ====================================================
    # Exibição das abas principais
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
    # 💬 Aba do Chat IA (com configuração e memória)
    # ====================================================
    with tabs[6]:
        st.header("🧠 Chat Inteligente com Memória Persistente")

        # ---- Seção de configuração da IA ----
        st.subheader("🔑 Configuração da API da IA")

        if "provider" not in st.session_state:
            st.session_state["provider"] = None
        if "user_api_key" not in st.session_state:
            st.session_state["user_api_key"] = ""

        col1, col2 = st.columns([1.5, 3])

        with col1:
            provider = st.selectbox(
                "Selecione o provedor de IA:",
                ["OpenAI", "Groq", "Gemini"],
                index=(
                    0
                    if not st.session_state["provider"]
                    else ["OpenAI", "Groq", "Gemini"].index(
                        st.session_state["provider"]
                    )
                ),
                key="provider_selector",
            )

        with col2:
            api_key = st.text_input(
                f"Insira sua API Key ({provider})",
                type="password",
                value=st.session_state.get("user_api_key", ""),
                key="api_key_input",
            )

        # ====================================================
        # ✅ SALVAR API KEY — sem redirecionar, sem recarregar
        # ====================================================
        if st.button("💾 Salvar Configuração de API"):
            st.session_state["provider"] = provider
            st.session_state["user_api_key"] = api_key
            st.success("✅ Configuração salva com sucesso!")

        st.divider()

        # ---- Chat em si ----
        render_chat(
            data=data,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            dataset_summary=st.session_state.get("dataset_summary"),
            api_key=st.session_state.get("user_api_key"),
            provider=st.session_state.get("provider"),
        )

    # ====================================================
    # Agora sim remove overlay — tudo foi carregado
    # ====================================================
    loading_container.empty()

else:
    st.info("👆 Envie um arquivo CSV para começar a análise.")
