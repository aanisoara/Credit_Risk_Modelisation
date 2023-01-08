import datetime
import pandas as pd


def get_year(x):
    return str(x)[:4]


def get_month(x):
    return str(x)[4:]


def get_date(x):
    return datetime.datetime(year=int(get_year(x)), month=int(get_month(x)), day=1)


def get_trimestre(x):
    if x.year == 2014:
        annee_var = 0
    elif x.year == 2015:
        annee_var = 1
    elif x.year == 2016:
        annee_var = 2
    elif x.year == 2017:
        annee_var = 3
    else:
        annee_var = 4
    return annee_var * 4 + (x.month - 1) // 3


def get_semestre(x):
    if x.year == 2014:
        annee_var = 0
    elif x.year == 2015:
        annee_var = 1
    elif x.year == 2016:
        annee_var = 2
    elif x.year == 2017:
        annee_var = 3
    else:
        annee_var = 4
    if x.month < 6:
        return 0 + annee_var * 2
    else:
        return 1 + annee_var * 2


def create_date_periods(data: pd.DataFrame, date_col: str = "date_debloc_avec_crd"):
    """
    Fonction qui créer les différentes périodes (trimestre, semestre et annuel) à partir de la date du dataframe

    Arguments:
        data -- Le dataset.
        date_col -- La colonne de date. Default to "date_debloc_avec_crd"

    Returns:
        Le dataset avec les colonnes "date", "trimestre", "semestre" et "annuel"
    """
    data["date"] = data[date_col].apply(get_date)
    data["trimestre"] = data["date"].apply(get_trimestre)
    data["semestre"] = data["date"].apply(get_semestre)
    data["annuel"] = data["date"].apply(lambda x: x.year)

    return data
