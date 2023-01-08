import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


def plot_stabilite_effectif(
    data: pd.DataFrame, colonnes: list, temporalite: str
) -> None:
    """
    Plot la stabilité temporelle en effectif

    Arguments:
        data -- (pd.DataFrame) : Le dataset
        colonnes -- (list) : La liste des colonnes du dataset à plot
        temporalite -- (str) : La colonne pour laquelle on veut analyser la temporalitée.
                                Recommandé : 'trimestre' ou 'semestre' ou 'annuel'
    """

    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(
        f"Stabilité temporelle ({temporalite}) en effectif des variables (base 100 en 2014)",
        fontsize=14,
        fontweight="bold",
        y=0.95,
    )

    ncols = 3
    nrows = len(colonnes) // ncols + (len(colonnes) % ncols > 0)

    for n, variable in enumerate(colonnes):
        # add a new subplot iteratively
        ax = plt.subplot(nrows, ncols, n + 1)

        modalites = data[variable].unique()
        for modalite in modalites:
            ax.plot(
                data[data[variable] == modalite][[variable, temporalite]]
                .groupby(by=temporalite)
                .count()
                / data[data[variable] == modalite][[variable, temporalite]]
                .groupby(by=temporalite)
                .count()
                .iloc[0]
                * 100,
                alpha=0.8,
                marker="o",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=2,
            )

        ax.set_xticks([])
        ax.set_title(variable.upper())


def plot_stabilite_risque(
    data: pd.DataFrame, target: np.ndarray, colonnes: list, temporalite: str
) -> None:
    """
    Plot la stabilité temporelle en risque

    Arguments:
        data -- (pd.DataFrame) : Le dataset
        target -- (np.ndarray) : Le vecteur binaire contenant les défauts
        colonnes -- (list) : La liste des colonnes du dataset à plot
        temporalite -- (str) : La colonne pour laquelle on veut analyser la temporalitée.
                                Recommandé : 'trimestre' ou 'semestre' ou 'annuel'
    """

    copied_data = data.copy()
    copied_data["target"] = target

    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)

    plt.suptitle(
        f"Stabilité temporelle ({temporalite}) en risque des variables",
        fontsize=14,
        fontweight="bold",
        y=0.95,
    )

    ncols = 3
    nrows = len(colonnes) // ncols + (len(colonnes) % ncols > 0)

    for n, variable in enumerate(colonnes):
        # add a new subplot iteratively
        ax = plt.subplot(nrows, ncols, n + 1)

        modalites = copied_data[variable].unique()
        for modalite in modalites:
            ax.plot(
                copied_data[copied_data[variable] == modalite][["target", temporalite]]
                .groupby(by=temporalite)
                .mean(),
                alpha=0.8,
                marker="o",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=2,
            )

        ax.set_xticks([])
        ax.set_title(variable.upper())

    del copied_data
