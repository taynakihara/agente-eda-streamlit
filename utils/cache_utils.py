import streamlit as st


def cache_clear_button():
    """
    Exibe um botão para limpar o cache de dados e sessão.
    Essa função remove tanto o cache do Streamlit (cache_data)
    quanto variáveis armazenadas em st.session_state.
    """

    st.markdown("---")
    col1, col2 = st.columns([1, 5])

    with col1:
        if st.button("🧹 Limpar Cache"):
            st.cache_data.clear()
            st.session_state.clear()
            st.success("Cache e sessão limpos com sucesso! Recarregue o arquivo CSV.")

    with col2:
        if "file_hash" in st.session_state:
            st.caption(
                "✅ Dados carregados do cache — gráficos prontos sem recarregar."
            )
        else:
            st.caption("⚙️ Nenhum cache ativo no momento.")
