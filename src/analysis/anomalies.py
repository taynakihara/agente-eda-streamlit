import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from utils.plot_utils import apply_blue_theme
import pandas as pd


# Função cacheada
@st.cache_data
def analyze_and_plot_anomalies(data: pd.DataFrame, numeric_cols: list):
    """Realiza os cálculos de outliers e gera todos os boxplots, cacheados."""
    summaries = []
    plots = {}

    for col in numeric_cols:
        # --- 1. Cálculo de Outliers ---
        Q1, Q3 = data[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = ((data[col] < lower) | (data[col] > upper)).sum()

        # O Z-Score precisa de dados não-nulos
        z_outliers = (np.abs(stats.zscore(data[col].dropna())) > 3).sum()
        summaries.append(
            {"Variável": col, "Outliers IQR": outliers, "Z-Score": z_outliers}
        )

        # --- 2. Geração do Boxplot ---
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.boxplot(
            data[col].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor="#0099FF", alpha=0.6),
        )
        ax.set_title(col)
        plots[col] = fig

    return summaries, plots


def render(data, numeric_cols):
    st.header("⚠️ Anomalias")
    apply_blue_theme()

    if not numeric_cols:
        st.info("Sem variáveis numéricas.")
        return

    # CHAMA FUNÇÃO CACHEADA
    summaries, plots = analyze_and_plot_anomalies(data, numeric_cols)

    # Exibe a tabela de resumo
    st.dataframe(summaries)

    st.subheader("Boxplots")

    # Exibe todos os plots do cache
    for fig in plots.values():
        st.pyplot(fig)
        st.markdown(
            "<hr style='border:1px solid #1E90FF; margin:2rem 0;'>",
            unsafe_allow_html=True,
        )
