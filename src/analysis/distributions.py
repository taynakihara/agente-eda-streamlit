import streamlit as st
import matplotlib.pyplot as plt
from utils.plot_utils import apply_blue_theme


def render(data, numeric_cols, categorical_cols):
    st.header("📊 Distribuições")
    apply_blue_theme()

    if numeric_cols:
        st.subheader("Variáveis Numéricas")
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(14, 7))  # largura x altura em polegadas
            ax.hist(
                data[col].dropna(),
                bins=30,
                color="#007BFF",
                edgecolor="#FFFFFF",  # cor da borda (pode trocar para outra)
                linewidth=1.2,  # espessura da linha da borda
                alpha=0.85,
            )

            ax.set_title(col)
            st.pyplot(fig)
    st.markdown(
        "<hr style='border:1px solid #1E90FF; margin:2rem 0;'>", unsafe_allow_html=True
    )

    if categorical_cols:
        st.subheader("Variáveis Categóricas")
        for col in categorical_cols[:5]:
            st.bar_chart(data[col].value_counts().head(10))
