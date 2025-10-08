import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from utils.plot_utils import apply_blue_theme

def render(data, numeric_cols):
    st.header("📈 Tendências")
    apply_blue_theme()

    time_cols = [c for c in data.columns if any(k in c.lower() for k in ["date", "time", "year", "month"])]
    if not time_cols or not numeric_cols:
        st.info("Nenhuma coluna temporal/númerica detectada.")
        return

    time_col = st.selectbox("Coluna temporal:", time_cols)
    value_col = st.selectbox("Variável numérica:", numeric_cols)
    data_sorted = data.sort_values(time_col)

    fig, ax = plt.subplots(figsize=(14, 7))  # largura x altura em polegadas
    ax.plot(data_sorted[time_col], data_sorted[value_col], color="#0099FF")
    ax.set_title(f"Tendência: {value_col} ao longo de {time_col}")
    st.pyplot(fig)
    st.markdown("<hr style='border:1px solid #1E90FF; margin:2rem 0;'>", unsafe_allow_html=True)
