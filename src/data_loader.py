import pandas as pd
import streamlit as st
import hashlib
from io import StringIO


def _hash_file(file) -> str:
    """
    Gera um hash SHA256 do conteúdo do arquivo CSV.
    Isso permite cache baseado no conteúdo, não apenas no nome.
    """
    content = file.getvalue()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False)
def load_data(file=None):
    """
    Carrega um arquivo CSV e identifica colunas numéricas e categóricas.
    O resultado é armazenado em cache para evitar recarregamentos.
    """

    try:
        # Gera hash do conteúdo do arquivo
        file_hash = _hash_file(file)

        # Lê o arquivo CSV usando pandas
        file.seek(0)
        data = pd.read_csv(file)

        # Identifica colunas
        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = data.select_dtypes(exclude=["number"]).columns.tolist()

        st.session_state["file_hash"] = (
            file_hash  # Armazena hash para comparação futura
        )

        return data, numeric_cols, categorical_cols

    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None, [], []
