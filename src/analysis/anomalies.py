import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from utils.plot_utils import apply_blue_theme

def render(data, numeric_cols):
    st.header("⚠️ Anomalias")
    apply_blue_theme()

    if not numeric_cols:
        st.info("Sem variáveis numéricas.")
        return

    summaries = []
    for col in numeric_cols:
        Q1, Q3 = data[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = ((data[col] < lower) | (data[col] > upper)).sum()
        z_outliers = (np.abs(stats.zscore(data[col].dropna())) > 3).sum()
        summaries.append({"Variável": col, "Outliers IQR": outliers, "Z-Score": z_outliers})

    st.dataframe(summaries)

    st.subheader("Boxplots")
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(14, 7))  # largura x altura em polegadas
        ax.boxplot(data[col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor="#0099FF", alpha=0.6))
        ax.set_title(col)
        st.pyplot(fig)
        
        st.markdown("<hr style='border:1px solid #1E90FF; margin:2rem 0;'>", unsafe_allow_html=True)
