# DEPOIS (clustering.py)

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.plot_utils import apply_blue_theme
import numpy as np


# NOVO BLOCO (Função cacheada)
@st.cache_data
def run_kmeans_and_plot(data, numeric_cols, n_clusters):
    """Executa o K-Means e gera o plot uma única vez por combinação de dados/k."""
    X = data[numeric_cols].dropna()

    kmeans = KMeans(
        n_clusters=n_clusters, n_init=10, random_state=42
    )  # random_state para reprodutibilidade
    clusters = kmeans.fit_predict(X)

    # Criação do DataFrame de exibição (apenas as 5 primeiras linhas para não ser pesado)
    df_preview = data.copy()
    df_preview["Cluster"] = clusters

    # Criação do Plot Matplotlib
    fig, ax = plt.subplots(figsize=(14, 7))  # largura x altura em polegadas
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap="Blues", alpha=0.7)
    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=200,
        color="black",
        marker="x",
    )
    ax.set_title("Visualização de Clusters")

    return fig, df_preview[["Cluster"] + numeric_cols].head()


def render(data, numeric_cols):
    st.header("🧩 Análise de Clusters (K-Means)")
    apply_blue_theme()

    if len(numeric_cols) < 2:
        st.info("É necessário ao menos duas variáveis numéricas para clusterização.")
        return

    n_clusters = st.slider("Número de Clusters (k)", 2, 10, 3)

    # CHAMADA À FUNÇÃO CACHEADA
    fig, df_preview = run_kmeans_and_plot(data, numeric_cols, n_clusters)

    st.dataframe(df_preview)  # Exibe o dataframe retornado da função cacheada

    st.pyplot(fig)  # Exibe o plot retornado da função cacheada

    st.markdown(
        "<hr style='border:1px solid #1E90FF; margin:2rem 0;'>", unsafe_allow_html=True
    )
