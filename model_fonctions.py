
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import researchpy as rp
import scipy.stats as stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from pingouin import kruskal

from sklearn.metrics import roc_curve, auc
import seaborn as sns


def stepwise_selection(X, y,
                           initial_list=[],
                           threshold_in=0.01,
                           threshold_out = 0.05,
                           verbose=True):
        """ Perform a forward-backward feature selection
        based on p-value from statsmodels.api.OLS
        Arguments:
            X - pandas.DataFrame with candidate features
            y - list-like with the target
            initial_list - list of features to start with (column names of X)
            threshold_in - include a feature if its p-value < threshold_in
            threshold_out - exclude a feature if its p-value > threshold_out
            verbose - whether to print the sequence of inclusions and exclusions
        Returns: list of selected features
        Always set threshold_in < threshold_out to avoid infinite looping.
        See https://en.wikipedia.org/wiki/Stepwise_regression for the details
        """
        included = list(initial_list)
        while True:
            changed=False
            # forward step
            excluded = list(set(X.columns)-set(included))
            new_pval = pd.Series(index=excluded)
            for new_column in excluded:
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed=True
                if verbose:
                    print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

            # backward step
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            # use all coefs except intercept
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max() # null if pvalues is empty
            if worst_pval > threshold_out:
                changed=True
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                if verbose:
                    print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
            if not changed:
                break
        return included
    
def str_to_int(df: pd.DataFrame) -> pd.DataFrame:
    
    for i in range(df.shape[0]):

        if type(df['DEPARTEMENT_CRI'][i]) == str and df['DEPARTEMENT_CRI'][i] != 'NR':
            df['DEPARTEMENT_CRI'][i] = int(float(df['DEPARTEMENT_CRI'][i]))

def kruskal_test(df, liste_var_continu):
    colonne = []
    p_val = []
    for col in liste_var_continu:
        if kruskal(data=df, dv='defaut_36mois', between=col)['p-unc'].values[0] < 0.05:
            p_val.append(kruskal(data=df, dv='defaut_36mois', between=col)['p-unc'].values[0])
            colonne.append(col)
    return p_val, colonne


def chi2(liste_de_variables_dicho: list, df: pd.DataFrame, seuil: float):
    dicho_significative = []

    for col in liste_de_variables_dicho:
        crosstab, test_results, expected = rp.crosstab(df["defaut_36mois"], df[col],
                                                   test= "chi-square",
                                                   expected_freqs= True,
                                                   prop= "cell")
        if test_results['results'][1] < seuil:
            dicho_significative.append(col)
    return dicho_significative

def corr_entre_quanti(df: pd.DataFrame):
    corr = df.corr()

    # Create a mask for values above 90%
    # But also below 100% since it variables correlated with the same one
    mask = (df.corr() > 0.3) & (df.corr() < 1.0)
    high_corr = corr[mask]

    # Create a new column mask using any() and ~
    col_to_filter_out = ~high_corr[mask].any()

    # Apply new mask
    X_clean = df[high_corr.columns[col_to_filter_out]]

    # Visualize cleaned dataset
    return X_clean

def encode_par_tx_defaut(
    data: pd.DataFrame, colonne: str, target: np.ndarray
) -> pd.Series:
    """
    Encode une variable en fonction du taux de défaut de ses n modalités
    (0 pour la modalité au taux le plus faible et n pour la modalité au taux le plus élevé)

    Arguments:
        data -- (pd.DataFrame) : Le dataframe.
        colonne -- (str) : La colonne à encoder.
        target -- (iterable) : Le vecteur cible. La target et le dataframe doivent avoir les mêmes indexes.

    Returns:
        (pd.Series) : La colonne encodée.
        (dictionnaire) : Le dictionnaire permettant de mapper les features (pour le test set).
    """
    modalites = data[colonne].unique()
    encoder = dict()

    for modalite in modalites:
        encoder[modalite] = target[data[colonne] == modalite].mean()

    for i, key in enumerate(dict(sorted(encoder.items(), key=lambda x: x[1])).keys()):
        encoder[key] = i

    return data[colonne].map(encoder), encoder

sns.set_style(style = "darkgrid")

def plot_roc_curve(model, X, y):
    """
    Plot la courbe ROC et renvoies l'AUC associée

    Arguments:
        model -- Le modèle utilisé. Doit disposer d'une méthode "predict_proba"
        X -- La matrice de design
        y -- Le vecteur des observations

    Returns:
        L'Area Under Curve du modèle
    """
    probs = model.predict_proba(X)
    plt.figure(figsize = (8, 8))
    fpr, tpr, _ = roc_curve(y, probs[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='C0', lw=2, label='Modèle clf (auc = %0.4f)' % roc_auc)
    plt.fill_between(x = fpr, y1 = tpr, y2 = fpr, color = "C0", alpha = .2)
    plt.plot([0, 1], [0, 1], color='C1', lw=2, linestyle='--', label='Aléatoire (auc = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux faux positifs')
    plt.ylabel('Taux vrais positifs')
    plt.title('Courbe ROC', fontsize = 11, fontweight = "bold")
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc