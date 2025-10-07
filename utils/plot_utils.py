import matplotlib.pyplot as plt

def apply_blue_theme():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor": "#001F3F",
        "figure.facecolor": "#00264D",
        "axes.edgecolor": "#66B2FF",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white"
    })
