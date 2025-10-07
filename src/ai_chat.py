import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from groq import Groq


# ---------------------------------------------------------
# Função genérica para chamadas de API das diferentes LLMs
# ---------------------------------------------------------
def call_ai_api(api_choice, api_key, messages, model):
    """
    Roteia a chamada para a API correta com base na escolha do usuário.
    Retorna o texto gerado pela IA.
    """

    # --- OpenAI ---
    if api_choice == "OpenAI":
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(model=model, messages=messages)
        if hasattr(response, "choices") and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return "❌ Erro: resposta inesperada da API OpenAI."

    # --- Groq ---
    elif api_choice == "Groq":
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(model=model, messages=messages)
        if hasattr(response, "choices") and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return "❌ Erro: resposta inesperada da API Groq."

    # --- Gemini ---
    elif api_choice == "Gemini":
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(messages[-1]["content"])
        return response.text

    else:
        return "❌ API inválida selecionada."


# ---------------------------------------------------------
# Renderização do Chat IA (interface)
# ---------------------------------------------------------
def render_chat(data, numeric_cols, categorical_cols):
    """
    Renderiza a interface de chat com IA, mantendo o histórico na sessão.
    Essa função é independente do cache de dados e gráficos.
    """

    st.markdown("## 🤖 Chat com IA")

    # Inicializa estado da conversa se ainda não existir
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Seleção da API
    api_choice = st.selectbox("API:", ["Groq", "OpenAI", "Gemini"], index=0)

    # Chave de API e modelo
    api_key = st.text_input(f"Chave {api_choice}:", type="password")

    model = st.selectbox(
        "Modelo:",
        {
            "Groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            "OpenAI": ["gpt-3.5-turbo", "gpt-4o-mini"],
            "Gemini": ["gemini-1.5-pro"],
        }[api_choice],
    )

    st.markdown("---")

    # Campo de pergunta
    question = st.text_area("Pergunta:", key="user_input", height=100)

    if st.button("🚀 Enviar"):
        if not api_key:
            st.warning("⚠️ Insira a chave da API antes de enviar.")
            return
        if not question.strip():
            st.warning("Digite uma pergunta antes de enviar.")
            return

        # ---------------------------
        # Prompt-base dinâmico (system message)
        # ---------------------------
        base_prompt = """
        Você é um assistente de IA altamente inteligente e dinâmico, capaz de responder perguntas sobre qualquer assunto — 
        não apenas sobre os dados enviados. 

        Quando o usuário fizer perguntas relacionadas ao dataset, analise os dados e gere insights claros, visuais e explicativos.
        Quando o usuário fizer perguntas gerais (fora dos dados), responda normalmente, de forma objetiva e útil.
        Quando o usuário fizer perguntas técnicas, explique os conceitos de forma simples e didática.
        Quando o usuário fizer perguntas complexas, divida a resposta em etapas lógicas.
        Use exemplos práticos sempre que possível.
        Quando o usuário fizer perguntas referentes ao arquivo enviado, sempre responda com base nos dados do arquivo.
        Se o usuário fizer perguntas sobre colunas específicas, use os nomes exatos das colunas

        Seja didático, profissional e evite respostas genéricas. 
        Explique o raciocínio por trás das respostas quando fizer sentido.
        """

        # Contexto do dataset (caso exista)
        dataset_context = (
            f"O dataset contém {len(data)} linhas e {len(data.columns)} colunas.\n"
            f"Colunas numéricas: {numeric_cols}.\n"
            f"Colunas categóricas: {categorical_cols}."
        )

        # Monta o contexto inicial (system message)
        messages = [
            {"role": "system", "content": base_prompt + "\n\n" + dataset_context}
        ]

        # Adiciona histórico recente e a nova pergunta
        messages += st.session_state.chat_history[-4:]
        messages.append({"role": "user", "content": question})

        # Executa a chamada à API com indicador de carregamento
        with st.spinner("💬 Aguardando resposta da IA..."):
            try:
                response = call_ai_api(api_choice, api_key, messages, model)
                st.session_state.chat_history.append(
                    {"role": "user", "content": question}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.error(f"Erro ao consultar a API: {e}")
                return

    # Histórico de conversa
    if st.session_state.chat_history:
        st.markdown("### 💬 Histórico de Conversa")
        for msg in st.session_state.chat_history[-8:]:
            if msg["role"] == "user":
                st.markdown(f"**🧑 Você:** {msg['content']}")
            else:
                st.markdown(f"**🤖 IA:** {msg['content']}")

    # Botão para limpar o histórico
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.chat_history = []
        st.info("Histórico de conversa limpo.")
