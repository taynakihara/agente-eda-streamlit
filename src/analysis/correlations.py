import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plot_utils import apply_blue_theme

def render(data, numeric_cols):
    st.header("🔍 Correlações")
    apply_blue_theme()

    if len(numeric_cols) < 2:
        st.info("É necessário ao menos duas variáveis numéricas.")
        return

    corr = data[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 7))  # largura x altura em polegadas
    sns.heatmap(corr, cmap="Blues", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Matriz de Correlação")
    st.pyplot(fig)
    
    st.markdown("<hr style='border:1px solid #1E90FF; margin:2rem 0;'>", unsafe_allow_html=True)
