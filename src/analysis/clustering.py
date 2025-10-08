import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.plot_utils import apply_blue_theme
import numpy as np

def render(data, numeric_cols):
    st.header("🧩 Análise de Clusters (K-Means)")
    apply_blue_theme()

    if len(numeric_cols) < 2:
        st.info("É necessário ao menos duas variáveis numéricas para clusterização.")
        return

    n_clusters = st.slider("Número de Clusters (k)", 2, 10, 3)
    X = data[numeric_cols].dropna()

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    clusters = kmeans.fit_predict(X)

    data["Cluster"] = clusters
    st.dataframe(data[["Cluster"] + numeric_cols].head())

    fig, ax = plt.subplots(figsize=(14, 7))  # largura x altura em polegadas
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap="Blues", alpha=0.7)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               s=200, color="black", marker="x")
    ax.set_title("Visualização de Clusters")
    st.pyplot(fig)
    
    st.markdown("<hr style='border:1px solid #1E90FF; margin:2rem 0;'>", unsafe_allow_html=True)
