import streamlit as st
import warnings
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
import streamlit as st  # <-- Adicionar st. importado

# Este bloco impede que o warning "st.rerun() in callback is a no-op" apareça.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Funções de Callback para persistência do estado
def update_tab_index():
    """Atualiza o índice da aba ativa usando a chave do widget (tab_selector)"""
    active_label = st.session_state["tab_selector"]
    tab_labels = [
        "📊 Distribuições",
        "🔍 Correlações",
        "📈 Tendências",
        "📉 Variância",
        "⚠️ Anomalias",
        "🧩 Clusters",
        "🤖 Chat IA",
    ]
    st.session_state["active_tab_index"] = tab_labels.index(active_label)


# ====================================================
# Define o estado inicial da aba
# ====================================================
if "active_tab_index" not in st.session_state:
    st.session_state["active_tab_index"] = 0  # Inicia na Distribuições

# ====================================================
# LÓGICA DE FEEDBACK APÓS LIMPEZA DO CACHE (CORRIGIDA)
# ====================================================
# ESTE BLOCO ESTÁ NO NÍVEL CORRETO (APÓS DEFINIÇÃO DA ABA, MAS ANTES DA CONFIG)
if (
    "cache_cleared_success" in st.session_state
    and st.session_state["cache_cleared_success"]
):
    # Exibe a mensagem de sucesso
    st.success("✅ Cache e sessão limpos com sucesso! O aplicativo foi resetado.")
    # Reseta a flag para não aparecer novamente
    del st.session_state["cache_cleared_success"]
    # O app irá recarregar e o if uploaded_file: será False.

    # A ÚLTIMA PEÇA DO QUEBRA-CABEÇA: ZERAR O UPLOAD
    # Remove a entrada do arquivo carregado do uploader (se existir no state)
    if "uploaded_file" in st.session_state:
        del st.session_state["uploaded_file"]


# ===============================
# 🔧 Configurações iniciais
# ===============================
load_dotenv()
st.set_page_config(page_title="🤖 Análise com IA", layout="wide")

# Isso desativa o toast de aviso "Calling st.rerun() within a callback is a no-op."
st.markdown(
    """
    <style>
    /* Esconde o elemento 'toast' (onde o aviso aparece) */
    .stApp > div:nth-child(1) > div:nth-child(1) > header > div:nth-child(1) > div:nth-child(2) {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
# Garante que a chave exista para evitar erro
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = "initial_uploader"

uploaded_file = st.file_uploader(
    "📂 Envie seu arquivo CSV", type=["csv"], key=st.session_state["uploader_key"]
)

if uploaded_file:
    # ====================================================
    # Inicializa o container de loading ANTES da lógica condicional
    # para garantir que esteja no escopo.
    # ====================================================
    loading_container = st.empty()

    # ====================================================
    # 🌀 LOADING VISUAL CONDICIONAL E BLOQUEANTE
    # ====================================================
    if not st.session_state.get("file_hash") or st.session_state.get("is_loading"):
        st.session_state["is_loading"] = True

        # Renderiza o overlay de loading
        loading_container.markdown(
            """
            <style>
            /* CSS que bloqueia a tela e exibe o spinner */
            .stApp { pointer-events: none; }
            .loading-overlay {
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background-color: rgba(0, 0, 0, 0.85); display: flex;
                flex-direction: column; justify-content: center; align-items: center;
                z-index: 1000; color: white;
            }
            .spinner {
                border: 10px solid #f3f3f3; border-top: 10px solid #3498db;
                border-radius: 50%; width: 80px; height: 80px;
                animation: spin 2s linear infinite;
            }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            </style>
            <div class="loading-overlay">
                <div class="spinner"></div>
                <h2>Aguarde, processando...</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ----------------------------------------------------
    # AGORA processa os dados (load_data é cacheada)
    # ----------------------------------------------------
    data, numeric_cols, categorical_cols = load_data(uploaded_file)

    # ====================================================
    # GERAÇÃO DO SUMÁRIO PARA O CHAT IA
    # ====================================================
    if "dataset_summary" not in st.session_state:
        with st.spinner("🧠 Gerando sumário do dataset para o Chat IA..."):
            # ❗ CONFIRME SE SEU ARQUIVO DE CHAT ESTÁ AQUI
            from src.ai_chat import summarize_dataset

            # O sumário deve ser gerado UMA ÚNICA VEZ
            st.session_state["dataset_summary"] = summarize_dataset(data)

    # Remove o flag de loading após o carregamento pesado
    if "is_loading" in st.session_state:
        del st.session_state["is_loading"]

    # ====================================================
    # 🔄 Limpa histórico e cache de sessão ao carregar novo arquivo
    # ====================================================
    for key in ["chat_history", "dataset_summary", "memoria_carregada"]:
        if key in st.session_state:
            del st.session_state[key]

    st.success(
        f"✅ Arquivo carregado: {data.shape[0]} linhas, {data.shape[1]} colunas."
    )

    # CHAMADA CORRETA: Usa a função importada
    cache_clear_button()

    # ====================================================
    # Exibição das abas principais - COM PERSISTÊNCIA E CALLBACK
    # ====================================================
    tab_labels = [
        "📊 Distribuições",
        "🔍 Correlações",
        "📈 Tendências",
        "📉 Variância",
        "⚠️ Anomalias",
        "🧩 Clusters",
        "🤖 Chat IA",
    ]

    st.radio(
        "Selecione uma aba:",
        options=tab_labels,
        index=st.session_state.get("active_tab_index", 0),
        horizontal=True,
        label_visibility="collapsed",
        key="tab_selector",
        # CHAVE DA CORREÇÃO: Chama o callback para salvar o estado antes do rerun
        on_change=update_tab_index,
    )

    # O active_tab_label não é mais necessário aqui, pois o estado é gerenciado pelo callback.
    # Vamos usar st.session_state["active_tab_index"] para renderizar o conteúdo.

    # Renderização do conteúdo APENAS da aba ativa (agora usando o índice da sessão)
    active_index = st.session_state.get("active_tab_index", 0)

    if tab_labels[active_index] == "📊 Distribuições":
        distributions.render(data, numeric_cols, categorical_cols)
    elif tab_labels[active_index] == "🔍 Correlações":
        correlations.render(data, numeric_cols)
    elif tab_labels[active_index] == "📈 Tendências":
        trends.render(data, numeric_cols)
    elif tab_labels[active_index] == "📉 Variância":
        variance.render(data, numeric_cols)
    elif tab_labels[active_index] == "⚠️ Anomalias":
        anomalies.render(data, numeric_cols)
    elif tab_labels[active_index] == "🧩 Clusters":
        clustering.render(data, numeric_cols)
    elif tab_labels[active_index] == "🤖 Chat IA":
        # ====================================================
        # 💬 Conteúdo da Aba Chat IA
        # ====================================================
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
    # Agora sim remove overlay
    # ====================================================
    if "loading_container" in locals() and st.session_state.get("file_hash"):
        loading_container.empty()

else:
    st.info("👆 Envie um arquivo CSV para começar a análise.")
