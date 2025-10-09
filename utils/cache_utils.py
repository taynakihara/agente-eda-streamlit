import streamlit as st


# Funções de callback para garantir a limpeza do estado
def clear_state_and_caches():
    # 1. Limpa todos os caches do Streamlit
    st.cache_data.clear()
    st.cache_resource.clear()

    # 2. Limpa TODAS as variáveis da sessão e define o flag de sucesso
    keys_to_delete = list(st.session_state.keys())
    for key in keys_to_delete:
        del st.session_state[key]

    # Define a flag de sucesso para exibição APÓS o rerun
    st.session_state["cache_cleared_success"] = True

    # 3. Zera o uploader.
    # Esta chave é injetada no st.file_uploader pelo Streamlit.
    st.session_state["uploader_key"] = ""

    # 4. Força o recarregamento.
    st.rerun()


def cache_clear_button():
    """Botão para limpar cache, sessão e memória do app completamente."""
    # Renderiza o botão que chama a função de limpeza
    st.button(
        "🧹 Limpar Cache",
        key="clear_cache_button",
        on_click=clear_state_and_caches,  # Chama a limpeza e o rerun
    )
