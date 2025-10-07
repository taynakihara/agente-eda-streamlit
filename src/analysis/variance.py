import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def render(data, numeric_cols):
    """
    Exibe o gráfico de variância para cada coluna numérica do dataset.
    """

    st.markdown("## 📉 Análise de Variância")
    st.markdown(
        "A variância mede o quanto os valores de cada variável numérica se dispersam "
        "em relação à média. Valores mais altos indicam maior dispersão."
    )

    if not numeric_cols:
        st.warning("⚠️ Nenhuma coluna numérica encontrada no dataset.")
        return

    # Calcula variância padronizada (normalizada entre 0 e 1)
    normalized_data = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
    variances = normalized_data.var().sort_values(ascending=False)

    # Gráfico de barras horizontais
    fig, ax = plt.subplots(figsize=(10, max(4, len(variances) * 0.4)))

    y_pos = np.arange(len(variances))
    ax.barh(y_pos, variances, color="#4DA6FF", edgecolor="#FFFFFF", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variances.index, color="#FFFFFF")
    ax.invert_yaxis()

    ax.set_xlabel("Valor da Variância", color="#A7C7E7")
    ax.set_title("Variância das Variáveis Numéricas", color="#FFFFFF", fontsize=14)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.patch.set_facecolor("#001F3F")
    ax.set_facecolor("#002B5C")

    st.pyplot(fig)
    st.markdown("<hr style='border:1px solid #1E90FF; margin:2rem 0;'>", unsafe_allow_html=True)
