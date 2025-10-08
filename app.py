import streamlit as st
from src.data_loader import load_data
from src.analysis import distributions, correlations, trends, anomalies, clustering
from src.analysis import variance
from src.ai_chat import render_chat
from utils.cache_utils import cache_clear_button
import time
from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# Configurações Gerais
# -----------------------------
st.set_page_config(page_title=" 🤖 Análise Exploratória com IA", layout="wide")

# === Ajuste visual: espaçamento entre o topo (barra do Streamlit) e o conteúdo ===
st.markdown(
    """
    <style>
        /* Aplica margem apenas no conteúdo principal */
        .block-container {
            margin-top: 3.5rem; /* ajuste aqui conforme desejar (3rem, 4rem etc.) */
        }

        /* Mantém o topo (Deploy bar) colado */
        header, .stAppHeader {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        /* Opcional: leve espaçamento inferior para estética */
        .main {
            padding-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Estilo geral (antes do upload)
# -----------------------------
st.markdown(
    """
    <style>
        /* Fundo geral */
        body {
            background: linear-gradient(180deg, #001F3F, #003366, #004080, #0059b3);
            color: white;
        }

        /* Área principal da aplicação */
        .stApp {
            background: linear-gradient(180deg, #001F3F, #003366, #004080, #0059b3);
            color: white;
        }

        /* Centralizar todo o conteúdo e limitar a largura */
        .block-container {
            max-width: 900px;         /* Ajusta a largura máxima */
            margin: 0 auto;           /* Centraliza horizontalmente */
            padding-top: 2rem;        /* Espaçamento superior */
        }

        /* Centraliza os elementos dentro do container */
        section.main > div {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        /* Upload Box menor e centralizado */
        .stFileUploader {
            width: 70%;               /* Diminui o tamanho do bloco de upload */
            max-width: 600px;
            margin: 0 auto;
        }

        /* Caixa de informação (azul) mais compacta */
        .stAlert {
            max-width: 600px;
            margin: 1rem auto;
            border-radius: 10px;
            font-size: 0.9rem;
        }

        /* Título centralizado */
        h1 {
            text-align: center !important;
            font-size: 2.4rem !important;
        }

        /* Subtítulos (seções das abas) */
        h2, h3 {
            text-align: center;
            color: #A7C7E7;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Cabeçalho e upload
# -----------------------------
st.title("📊 Análise Exploratória de Dados com IA")

uploaded_file = st.file_uploader("📂 Envie seu arquivo CSV para análise", type=["csv"])

# -----------------------------
# Ajuste visual dinâmico (após upload)
# -----------------------------
if uploaded_file:
    st.markdown(
        """
        <style>
            /* Expande layout após o upload */
            .block-container {
                max-width: 95% !important;  /* ocupa quase toda a largura da tela */
                padding-left: 3%;
                padding-right: 3%;
            }

            /* Faz gráficos usarem a largura total */
            .stPlotlyChart, .stPyplot {
                width: 100% !important;
                max-width: 100% !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Lógica principal da aplicação
# -----------------------------
if uploaded_file:
    # --- Cria overlay de loading antes de renderizar qualquer gráfico ---
    loading_container = st.empty()
    loading_container.markdown(
        """
        <div id="loading-overlay">
            <div class="loading-content">
                <div class="spinner"></div>
                <p>Carregando gráficos e análises... aguarde um momento ⏳</p>
            </div>
        </div>

        <style>
            #loading-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 30, 0.97);
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
                width: 70px;
                height: 70px;
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

    # --- Carrega dados (com cache inteligente) ---
    data, numeric_cols, categorical_cols = load_data(uploaded_file)

    # --- Exibe status e botão de cache ---
    st.success(
        f"Arquivo carregado com sucesso! ({data.shape[0]} linhas, {data.shape[1]} colunas)"
    )
    cache_clear_button()

# ...existing code...

# --- Cria todas as abas, mas só exibe após carregamento completo ---
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

if uploaded_file:
    # --- Renderiza conteúdo dentro das abas ---
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

    # --- Aba do Chat IA (com memória persistente) ---
    with tabs[6]:
        st.header("🤖 Chat Inteligente com Memória")

        # --- Inicialização da memória do agente ---
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if "dataset_summary" not in st.session_state and data is not None:
            from src.ai_chat import summarize_dataset

            st.session_state["dataset_summary"] = summarize_dataset(data)

        # --- Botão para limpar memória do agente ---
        if st.button("🧹 Limpar memória do agente"):
            st.session_state["chat_history"] = []
            st.session_state["dataset_summary"] = None
            st.success("Memória do agente limpa!")

        # --- Configuração de API ---
        st.subheader("🔑 Configuração da API da IA")
        if "provider" not in st.session_state:
            st.session_state["provider"] = "OpenAI"
        if "user_api_key" not in st.session_state:
            st.session_state["user_api_key"] = ""

        # Esses widgets NÃO devem recarregar a página
        provider = st.selectbox(
            "Selecione o provedor de IA:",
            ["OpenAI", "Groq", "Gemini"],
            key="provider_selector",
            index=["OpenAI", "Groq", "Gemini"].index(st.session_state["provider"]),
            on_change=None,  # evita reexecução desnecessária
        )
        api_key = st.text_input(
            f"Insira sua API Key ({provider})",
            type="password",
            value=st.session_state["user_api_key"],
            key="user_api_key_input",
            on_change=None,  # evita recarregar a página
        )

        # Armazena apenas quando o usuário confirmar
        if st.button("💾 Salvar Configuração de API"):
            st.session_state["provider"] = provider
            st.session_state["user_api_key"] = api_key
            st.success(f"✅ Configuração salva: {provider}")

        # --- Renderização do chat com memória contextual ---
        from src.ai_chat import render_chat

        render_chat(
            data,
            numeric_cols,
            categorical_cols,
            st.session_state["dataset_summary"],
            api_key=api_key,
            provider=provider,
        )

    # --- Remove o loading SOMENTE após todas as abas renderizarem ---
    st.session_state["loaded"] = True
    loading_container.empty()
else:
    st.info("👆 Carregue um arquivo CSV para começar a análise.")
# ...existing code...
