import streamlit as st
import matplotlib.pyplot as plt
from utils.plot_utils import apply_blue_theme
import pandas as pd  # <-- Adicionado para tipagem, se necessário


# Função cacheada para Histograms - Plots Numéricos
@st.cache_data
def generate_numeric_histograms(data: pd.DataFrame, numeric_cols: list):
    """Gera e armazena em cache todos os gráficos de distribuição numérica."""
    plots = {}
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.hist(
            data[col].dropna(),
            bins=30,
            color="#007BFF",
            edgecolor="#FFFFFF",
            linewidth=1.2,
            alpha=0.85,
        )
        ax.set_title(col)
        plots[col] = fig
    return plots


# Função cacheada para Bar Charts
@st.cache_data
def generate_categorical_bar_charts(data: pd.DataFrame, categorical_cols: list):
    """Gera e armazena em cache todos os gráficos de distribuição categórica."""
    charts = {}
    for col in categorical_cols[:5]:  # Mantendo o limite de 5
        # st.bar_chart usa o objeto ValueCounts, que é seguro para cache.
        charts[col] = data[col].value_counts().head(10)
    return charts


def render(data, numeric_cols, categorical_cols):
    st.header("📊 Distribuições")
    apply_blue_theme()

    if numeric_cols:
        st.subheader("Variáveis Numéricas")

        # CHAMA FUNÇÃO CACHEADA E EXIBE PLOTS
        numeric_plots = generate_numeric_histograms(data, numeric_cols)
        for fig in numeric_plots.values():
            st.pyplot(fig)  # AGORA EXIBE O CACHE INSTANTANEAMENTE

    st.markdown(
        "<hr style='border:1px solid #1E90FF; margin:2rem 0;'>", unsafe_allow_html=True
    )

    if categorical_cols:
        st.subheader("Variáveis Categóricas")

        # CHAMA FUNÇÃO CACHEADA E EXIBE CHARTS
        categorical_charts = generate_categorical_bar_charts(data, categorical_cols)
        for col, chart_data in categorical_charts.items():
            st.write(f"#### {col}")  # Adiciona um título para cada gráfico de barras
            st.bar_chart(chart_data)
